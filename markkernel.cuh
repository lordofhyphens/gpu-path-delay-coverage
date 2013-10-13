#include "markkernel.h"
#include "util/gpudata.h"
#include "util/utility.cuh"
#undef LOGEXEC
#include <cuda.h>
#undef MARK_BLOCK
#define MARK_BLOCK 128
#define INV_MARK_BLOCK 128
#undef OUTJUST
#define OUTJUST 4
#define BLOCK_PER_KERNEL 8
void HandleMarkError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleMarkError( err, __FILE__, __LINE__ ))

#define LUT_16(TYP,C,T) (((TYP) << (C*4+T)&0x8000) >> 15)

// tiny constant-memory lookup table
__device__ __constant__ __uint16_t LUTs[6] = { 0x0054, 0x0054, 0x008A, 0x008A, 0x0000,0x0000 };

DEVICE coalesce_t robust(const uint8_t f1, const uint8_t *sim, const size_t& sim_pitch, const uint32_t& tid, const uint32_t *fans,const GPUNODE& gate, const coalesce_t resultCache, const coalesce_t oldmark) {
	coalesce_t tmp; tmp.packed = 0xFFFFFFFF;
	const coalesce_t sim_fin1 = REF2D((coalesce_t*)sim,sim_pitch,tid,FIN(fans,gate.offset,f1));
	for (uint16_t fin2 = 0; fin2 < gate.nfi; fin2++) {
		if (f1 == fin2) continue;
		const coalesce_t sim_fin2 = REF2D((coalesce_t*)sim,sim_pitch,tid,FIN(fans,gate.offset,fin2));
		tmp.rows[0] = tmp.rows[0] && LUT_16(LUTs[gate.type-2],sim_fin1.rows[0], sim_fin2.rows[0]);
		tmp.rows[1] = tmp.rows[1] && LUT_16(LUTs[gate.type-2],sim_fin1.rows[1], sim_fin2.rows[1]);
		tmp.rows[2] = tmp.rows[2] && LUT_16(LUTs[gate.type-2],sim_fin1.rows[2], sim_fin2.rows[2]);
		tmp.rows[3] = tmp.rows[3] && LUT_16(LUTs[gate.type-2],sim_fin1.rows[3], sim_fin2.rows[3]);
	}
	tmp.packed = tmp.packed & oldmark.packed;
	tmp.packed = tmp.packed | resultCache.packed;
	return tmp;
}
extern "C" __launch_bounds__(MARK_BLOCK,BLOCK_PER_KERNEL) 
__global__ void kernMarkPathSegments(uint8_t *sim, size_t sim_pitch, uint8_t* mark, size_t pitch, size_t patterns, GPUNODE* node, uint32_t* fans, int start, int startPattern) {
	int tid = (blockIdx.y * blockDim.x) + threadIdx.x;
	int gid = (blockIdx.x) + start;
	coalesce_t resultCache; resultCache.packed = 0;
	if ((tid*4) < patterns) {
		const GPUNODE& gate = node[gid];
		const coalesce_t simResults = REF2D((coalesce_t*)sim,sim_pitch,tid,gid);
		coalesce_t rowCache((simResults.rows[0] > 1), (simResults.rows[1] > 1), (simResults.rows[2] > 1),(simResults.rows[3] > 1));

		// switching based on value causes divergence, switch based on node type instead.
		switch(gate.type) {
			case INPT:
				if (gate.nfo == 0 && gate.po < 1) {
					resultCache.packed = 0; // on the odd case that an input is literally connected to nothing, this is not a path.
				} else {
					resultCache = rowCache;// Otherwise we can mark this.
				}
				break;
			case FROM: // For FROM, it's equal to its fan-in
			case BUFF:
			case DFF:
				// For inverter and buffer gates, mark if and only if a fan-in is marked.
			case NOT: resultCache = REF2D((coalesce_t*)mark,pitch,tid,FIN(fans,gate.offset,0)); break;
			case OR:  // For the normal gates, set the fan-out based on the fan-ins. 
			case NOR: // There's a LUT for each basic gate type.
			case XOR:
			case XNOR:
			case NAND:
			case AND:
				for (uint16_t fin1 = 0; fin1 < gate.nfi; fin1++) {
					coalesce_t oldmark = REF2D((coalesce_t*)mark,pitch,tid,FIN(fans,gate.offset,fin1));
					resultCache = robust(fin1, sim, sim_pitch, tid, fans, gate,resultCache, oldmark);
				}
				break;
			default: break;
		}
		// stick the contents of resultCache into the mark array
		resultCache.packed = resultCache.packed & rowCache.packed;
		REF2D((coalesce_t*)mark,pitch,tid,gid) = resultCache;
	}
}
extern "C" __launch_bounds__(INV_MARK_BLOCK,BLOCK_PER_KERNEL) 
__global__ void kernRemoveInvalidMarks(uint8_t* mark, size_t pitch, size_t patterns, GPUNODE* node, uint32_t* fans, int start) {
	int tid = (blockIdx.y * blockDim.x) + threadIdx.x;
	int gid = (blockIdx.x) + start;
	if ((tid*4) < patterns) {
		const GPUNODE& gate = node[gid];
		// there must be at least one fan-out of current gate that is marked, if not a po
		coalesce_t result;
		result.packed = (gate.po > 0) * 0x01010101;;
		for (uint16_t j = 0; j < gate.nfo; j++) {
			result.packed = (result.packed | REF2D((coalesce_t*)mark, pitch, tid, FIN(fans,gate.offset,j+gate.nfi)).packed);
		}
		result.packed = result.packed & REF2D((coalesce_t*)mark,pitch,tid,gid).packed;
		REF2D((coalesce_t*)mark,pitch,tid,gid) = result;
	}
}

float gpuMarkPaths(GPU_Data& results, GPU_Data& input, GPU_Circuit& ckt, size_t chunk, size_t startPattern) {
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting before we hit this point.
	int blockcount_y;
	int startGate;
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	blockcount_y = (int)(results.gpu(chunk).width/(MARK_BLOCK/4)) + ((results.gpu(chunk).width% (MARK_BLOCK/4)) > 0);
	startGate=0;
	for (uint32_t i = 0; i <= ckt.levels(); i++) {
		int levelsize = ckt.levelsize(i);
		do { 
			int simblocks = min(MAX_BLOCKS, levelsize);
			dim3 numBlocks(simblocks,blockcount_y);
			kernMarkPathSegments<<<numBlocks,MARK_BLOCK>>>(input.gpu(chunk).data, input.gpu(chunk).pitch, results.gpu(chunk).data, results.gpu(chunk).pitch, results.gpu(chunk).width,ckt.gpu_graph(), ckt.offset(),  startGate, startPattern);
			startGate += simblocks;
			if (levelsize > MAX_BLOCKS) {
				levelsize -= MAX_BLOCKS;
			} else {
				levelsize = 0;
			}
		} while (levelsize > 0);
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
	}
	if (verbose_flag != 0 && mark_flag != 0)
		debugMarkOutput(&results, ckt, chunk, startPattern, "gpumark-full.log");
	// do a backwards traversal to ensure that invalid marks are cleared.
	startGate=ckt.size();
	for (uint32_t i2 = 0; i2 <= ckt.levels(); i2++) {
		uint32_t i = ckt.levels() - i2;
		int levelsize = ckt.levelsize(i);
		do { 
			int simblocks = min(MAX_BLOCKS, levelsize);
			dim3 numBlocks(simblocks,blockcount_y);
			startGate -= simblocks;
			kernRemoveInvalidMarks<<<numBlocks,INV_MARK_BLOCK>>>(results.gpu(chunk).data, results.gpu(chunk).pitch, results.gpu(chunk).width,ckt.gpu_graph(), ckt.offset(),  startGate);
			if (levelsize > MAX_BLOCKS) {
				levelsize -= MAX_BLOCKS;
			} else {
				levelsize = 0;
			}
		} while (levelsize > 0);
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
	}
	cudaDeviceSynchronize();
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif // NTIMING
	if (verbose_flag != 0 && mark_flag != 0)
		debugMarkOutput(&results, ckt, chunk, startPattern, "gpumark.log");
#ifndef NTIMING
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}

void debugMarkOutput(GPU_Data* results, const GPU_Circuit& ckt,const size_t chunk, const size_t startPattern,std::string outfile = "simdebug.log") {
#ifndef NDEBUG
	std::ofstream ofile;
	size_t t = 0;
	if (chunk == 0) {
		ofile.open(outfile.c_str(), std::ios::out);
		ofile << "Gate:     " << "\t";
		for (uint32_t i = 0; i < ckt.size(); i++) {
			ofile << std::setw(OUTJUST) << i << " ";
		}
		ofile << "\n";
	} else {
		ofile.open(outfile.c_str(), std::ios::app);
	}

		uint8_t *lvalues;
		lvalues = (uint8_t*)malloc(results->gpu(chunk).height*results->gpu(chunk).pitch);
		cudaMemcpy2D(lvalues,results->gpu().pitch,results->gpu(chunk).data,results->gpu(chunk).pitch,results->gpu(chunk).width,results->gpu(chunk).height,cudaMemcpyDeviceToHost);
		for (unsigned int r = 0;r < results->gpu(chunk).width; r++) {
			ofile << "Vector " << t+startPattern << ":\t";
			for (unsigned int i = 0; i < results->gpu(chunk).height; i++) {
				uint8_t z = REF2D(lvalues, results->gpu(chunk).pitch, r, i);
				switch(z) {
					case 0:
						ofile  << std::setw(OUTJUST) << "N" << " "; break;
					case 1:
						ofile  << std::setw(OUTJUST) << "Y" << " "; break;
					default:
						ofile << std::setw(OUTJUST) << (int)z << " "; break;
				}
			}
			ofile << "\n";
			t++;
		}
		free(lvalues);
	ofile.close();
#endif

}
void debugMarkOutput(GPU_DATA_type<coalesce_t>* results, const GPU_Circuit& ckt,const size_t chunk, const size_t startPattern,std::string outfile = "simdebug.log") {
#ifndef NDEBUG
	std::ofstream ofile;
	size_t t = 0;
	if (chunk == 0) {
		ofile.open(outfile.c_str(), std::ios::out);
		ofile << "Gate:     " << "\t";
		for (uint32_t i = 0; i < ckt.size(); i++) {
			ofile << std::setw(OUTJUST) << i << " ";
		}
		ofile << "\n";
	} else {
		ofile.open(outfile.c_str(), std::ios::app);
	}

		uint8_t *lvalues;
		lvalues = (uint8_t*)malloc(results->height*results->pitch);
		cudaMemcpy2D(lvalues,results->pitch,results->data,results->pitch,results->width,results->height,cudaMemcpyDeviceToHost);
		for (unsigned int r = 0;r < results->width; r++) {
			ofile << "Vector " << t+startPattern << ":\t";
			for (unsigned int i = 0; i < results->height; i++) {
				uint8_t z = REF2D(lvalues, results->pitch, r, i);
				switch(z) {
					case 0:
						ofile  << std::setw(OUTJUST) << "N" << " "; break;
					case 1:
						ofile  << std::setw(OUTJUST) << "Y" << " "; break;
					default:
						ofile << std::setw(OUTJUST) << (int)z << " "; break;
				}
			}
			ofile << "\n";
			t++;
		}
		free(lvalues);
	ofile.close();
#endif

}

