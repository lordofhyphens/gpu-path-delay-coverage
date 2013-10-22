#include "util/defines.h"
#include "util/gpudata.h"
#include "util/utility.cuh"
#undef LOGEXEC
#include <cuda.h>
#undef MARK_BLOCK
#define MARK_BLOCK 128
#define INV_MARK_BLOCK 128
#undef OUTJUST
#define OUTJUST 4
#undef BLOCK_PER_KERNEL
#define BLOCK_PER_KERNEL 8
void HandleMarkError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#undef HANDLE_ERROR
#define HANDLE_ERROR( err ) (HandleMarkError( err, __FILE__, __LINE__ ))

#include <cassert>
#include <stdint.h>
#include "util/defines.h"
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/utility.h"
#include "util/segment.cuh"
#define DEVICE __device__ __forceinline__

float gpuMarkPaths(GPU_Data& results, GPU_Data& input, GPU_Circuit& ckt, size_t chunk, size_t startPattern);
void debugMarkOutput(ARRAY2D<uint8_t> results, const GPU_Circuit& ckt, std::string outfile = "markdebug.log");
void debugMarkOutput(GPU_Data* results, const GPU_Circuit& ckt,const size_t chunk, const size_t startPattern,std::string outfile);
void debugMarkOutput(GPU_DATA_type<coalesce_t>* results, const GPU_Circuit& ckt,const size_t chunk, const size_t startPattern,std::string outfile);
// tiny constant-memory lookup table
__device__ __constant__ __uint16_t LUT_NR[6] = { 0x2A00, 0x2A00, 0x5100, 0x5100, 0x0000,0x0000 };
__device__ __constant__ __uint16_t LUT_R[6] = { 0x2200, 0x2200, 0x1100, 0x1100, 0x0000,0x0000 };

__device__ __forceinline__ uint8_t LUT_16(const uint16_t type, const uint16_t on, const uint16_t off) {
	return (type >> ((on << 2)+off)) & 0x0001;
}

__device__ __forceinline__  coalesce_t robust(const uint8_t f1, const uint8_t *sim, const size_t sim_pitch, const uint32_t& tid, const uint32_t *fans,const GPUNODE& gate, const coalesce_t resultCache, const coalesce_t oldmark) {
	coalesce_t tmp; tmp.packed = 0xFFFFFFFF;
	const coalesce_t sim_fin1 = REF2D((coalesce_t*)sim,sim_pitch,tid,FIN(fans,gate.offset,f1));
	for (uint16_t fin2 = 0; fin2 < gate.nfi; fin2++) {
		if (f1 == fin2) continue;
		const coalesce_t sim_fin2 = REF2D((coalesce_t*)sim,sim_pitch,tid,FIN(fans,gate.offset,fin2));
		tmp.rows[0] = tmp.rows[0] && LUT_16(LUT_R[gate.type-2],sim_fin1.rows[0], sim_fin2.rows[0]);
		tmp.rows[1] = tmp.rows[1] && LUT_16(LUT_R[gate.type-2],sim_fin1.rows[1], sim_fin2.rows[1]);
		tmp.rows[2] = tmp.rows[2] && LUT_16(LUT_R[gate.type-2],sim_fin1.rows[2], sim_fin2.rows[2]);
		tmp.rows[3] = tmp.rows[3] && LUT_16(LUT_R[gate.type-2],sim_fin1.rows[3], sim_fin2.rows[3]);
	}
	tmp.packed = tmp.packed & oldmark.packed;
	tmp.packed = tmp.packed | resultCache.packed;
	return tmp;
}
__device__ __forceinline__  coalesce_t nonrobust(const uint8_t f1, const uint8_t *sim, const size_t sim_pitch, const uint32_t& tid, const uint32_t *fans,const GPUNODE& gate, const coalesce_t resultCache, const coalesce_t oldmark) {
	coalesce_t tmp; tmp.packed = 0xFFFFFFFF;
	const coalesce_t sim_fin1 = REF2D((coalesce_t*)sim,sim_pitch,tid,FIN(fans,gate.offset,f1));
	for (uint16_t fin2 = 0; fin2 < gate.nfi; fin2++) {
		if (f1 == fin2) continue;
		const coalesce_t sim_fin2 = REF2D((coalesce_t*)sim,sim_pitch,tid,FIN(fans,gate.offset,fin2));
		tmp.rows[0] = tmp.rows[0] && LUT_16(LUT_NR[gate.type-2],sim_fin1.rows[0], sim_fin2.rows[0]);
		tmp.rows[1] = tmp.rows[1] && LUT_16(LUT_NR[gate.type-2],sim_fin1.rows[1], sim_fin2.rows[1]);
		tmp.rows[2] = tmp.rows[2] && LUT_16(LUT_NR[gate.type-2],sim_fin1.rows[2], sim_fin2.rows[2]);
		tmp.rows[3] = tmp.rows[3] && LUT_16(LUT_NR[gate.type-2],sim_fin1.rows[3], sim_fin2.rows[3]);
	}
	tmp.packed = tmp.packed & oldmark.packed;
	tmp.packed = tmp.packed | resultCache.packed;
	return tmp;
}

__device__ __forceinline__  coalesce_t robust(const uint32_t f1, const uint8_t *sim, const size_t sim_pitch, const uint32_t tid, const uint32_t *fans,const GPUNODE& gate) {
	coalesce_t tmp; tmp.packed = 0xFFFFFFFF;
	const coalesce_t sim_fin1 = REF2D((coalesce_t*)sim,sim_pitch,tid,f1);
	switch(gate.type){
		case FROM: // For FROM, it's equal to its fan-in
		case BUFF:
		case DFF:
			// For inverter and buffer gates, mark if and only if a fan-in is marked.
		case NOT: tmp = 0x01010101;break;
		case OR:  // For the normal gates, set the fan-out based on the fan-ins. 
		case NOR: // There's a LUT for each basic gate type.
		case XOR:
		case XNOR:
		case NAND:
		case AND:
			for (uint16_t fin2 = 0; fin2 < gate.nfi; fin2++) {
				if (f1 == FIN(fans,gate.offset,fin2)) continue;
				const coalesce_t sim_fin2 = REF2D((coalesce_t*)sim,sim_pitch,tid,FIN(fans,gate.offset,fin2));
				tmp.rows[0] = tmp.rows[0] && LUT_16(LUT_R[gate.type-2],sim_fin1.rows[0], sim_fin2.rows[0]);
				tmp.rows[1] = tmp.rows[1] && LUT_16(LUT_R[gate.type-2],sim_fin1.rows[1], sim_fin2.rows[1]);
				tmp.rows[2] = tmp.rows[2] && LUT_16(LUT_R[gate.type-2],sim_fin1.rows[2], sim_fin2.rows[2]);
				tmp.rows[3] = tmp.rows[3] && LUT_16(LUT_R[gate.type-2],sim_fin1.rows[3], sim_fin2.rows[3]);
			} break;
	}
	return tmp;

}
__device__ __forceinline__  coalesce_t nonrobust(const uint32_t f1, const uint8_t *sim, const size_t sim_pitch, const uint32_t& tid, const uint32_t *fans,const GPUNODE gate) {
	coalesce_t tmp; tmp.packed = 0xFFFFFFFF;
	const coalesce_t sim_fin1 = REF2D((coalesce_t*)sim,sim_pitch,tid,f1);
	switch(gate.type){
		case FROM: // For FROM, it's equal to its fan-in
		case BUFF:
		case DFF:
			// For inverter and buffer gates, mark if and only if a fan-in is marked.
		case NOT: tmp = 0x01010101;break;
		case OR:  // For the normal gates, set the fan-out based on the fan-ins. 
		case NOR: // There's a LUT for each basic gate type.
		case XOR:
		case XNOR:
		case NAND:
		case AND:
			for (uint16_t fin2 = 0; fin2 < gate.nfi; fin2++) {
				if (f1 == FIN(fans,gate.offset,fin2)) continue;
				const coalesce_t sim_fin2 = REF2D((coalesce_t*)sim,sim_pitch,tid,FIN(fans,gate.offset,fin2));
				tmp.rows[0] = tmp.rows[0] && LUT_16(LUT_NR[gate.type-2],sim_fin1.rows[0], sim_fin2.rows[0]);
				tmp.rows[1] = tmp.rows[1] && LUT_16(LUT_NR[gate.type-2],sim_fin1.rows[1], sim_fin2.rows[1]);
				tmp.rows[2] = tmp.rows[2] && LUT_16(LUT_NR[gate.type-2],sim_fin1.rows[2], sim_fin2.rows[2]);
				tmp.rows[3] = tmp.rows[3] && LUT_16(LUT_NR[gate.type-2],sim_fin1.rows[3], sim_fin2.rows[3]);
			} break;
	}
	return tmp;
}
__device__ void printGate(const int gid, const int in_gid, const GPUNODE gate, const uint8_t* sim, const size_t sim_pitch, const uint8_t mark, const int tid, uint32_t* fans) {
	int in_tid = (blockIdx.y * blockDim.x) + threadIdx.x;
	if (in_tid == tid && gid == in_gid) {
		printf("Gate particulars for %d:", gid);
		switch (gate.type) {
			case FROM:
				printf("FROM. %d NFI, %d NFO\n", gate.nfi, gate.nfo);
				break;
			case INPT:
				printf("INPT. %d NFI, %d NFO\n", gate.nfi, gate.nfo);
				break;
			case BUFF:
				printf("BUFF. %d NFI, %d NFO\n", gate.nfi, gate.nfo);
				break;
			case NOT:
				printf("NOT. %d NFI, %d NFO\n", gate.nfi, gate.nfo);
				break;
			case AND:
				printf("AND. %d NFI, %d NFO\n", gate.nfi, gate.nfo);
				break;
			case NAND:
				printf("NAND. %d NFI, %d NFO\n", gate.nfi, gate.nfo);
				break;
			case NOR:
				printf("NOR. %d NFI, %d NFO\n", gate.nfi, gate.nfo);
				break;
			case OR:
				printf("OR. %d NFI, %d NFO\n", gate.nfi, gate.nfo);
				break;
			case XOR:
				printf("XOR. %d NFI, %d NFO\n", gate.nfi, gate.nfo);
				break;
			case XNOR:
				printf("XNOR. %d NFI, %d NFO\n", gate.nfi, gate.nfo);
				break;
		}
		printf("Simulation for fan-ins (");
		for (int i = 0; i < gate.nfi; i++) {
			printf("%d", FIN(fans,gate.offset,i));
			if (i+1 < gate.nfi) {
				printf(", ");
			}
		}
		printf("): ");
		for (int i = 0; i < gate.nfi; i++) {
			int sim1 = REF2D((coalesce_t*)sim, sim_pitch,tid, FIN(fans,gate.offset,i)).rows[0];
			switch (sim1) {
				case T0:
					printf("T0"); break;
				case S1:
					printf("S1"); break;
				case T1:
					printf("T1"); break;
				case S0:
					printf("S0"); break;
				default:
					printf("Garbage");break;
			}
			if (i+1 < gate.nfi) {
				printf(", ");
			}
		}
		printf("\n");
		printf("Simulation result (tid %d): ",tid);
		uint8_t gsim = REF2D((coalesce_t*)sim, sim_pitch,tid, gid).rows[0];
		switch (gsim) {
			case T0:
				printf("T0\n"); break;
			case S1:
				printf("S1\n"); break;
			case T1:
				printf("T1\n"); break;
			case S0:
				printf("S0\n"); break;
			default:
				printf("Garbage.\n");break;
		}
		printf("Marked?: ");
		switch (mark) {
			case 0:
				printf("No.\n");break;
			case 1:
				printf("Yes.\n");break;
			default:
				printf("Garbage.\n");break;
		}
	}
}
__global__ void kernMarkPathSegments(uint8_t *sim, size_t sim_pitch, uint8_t* mark, size_t pitch, size_t patterns, GPUNODE* node, uint32_t* fans, int start, int startPattern, int robust_f) {
	int tid = (blockIdx.y * blockDim.x) + threadIdx.x;
	int gid = (blockIdx.x) + start;
	coalesce_t resultCache; resultCache.packed = 0;
	if ((tid*4) < patterns) {
		const GPUNODE gate = node[gid];
		const coalesce_t simResults = REF2D((coalesce_t*)sim,sim_pitch,tid,gid);
		coalesce_t rowCache((simResults.rows[0] > 1), (simResults.rows[1] > 1), (simResults.rows[2] > 1),(simResults.rows[3] > 1));

		// switching based on value causes divergence, switch based on node type instead.
		switch(gate.type) {
			case INPT:
				if (gate.nfo == 0 && gate.po < 1) {
					//resultCache.packed = 0; // on the odd case that an input is literally connected to nothing, this is not a path.
					resultCache = rowCache;// Otherwise we can mark this if it has a transition.
				} else {
					resultCache = rowCache;// Otherwise we can mark this if it has a transition.
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
					  if (robust_f == 1) {
						  resultCache = robust(fin1, sim, sim_pitch, tid, fans, gate,resultCache, oldmark); 
					  } else {

						  resultCache = nonrobust(fin1, sim, sim_pitch, tid, fans, gate,resultCache, oldmark);
					  }
				  }
				  break;
			default: break;
		}
		// stick the contents of resultCache into the mark array
		resultCache.packed = resultCache.packed & rowCache.packed;
		REF2D((coalesce_t*)mark,pitch,tid,gid) = resultCache;
		//printGate(1163, gid, gate, sim, sim_pitch, REF2D((coalesce_t*)mark,pitch,tid,gid).rows[0], 0, fans);
	}
}
extern "C" __launch_bounds__(INV_MARK_BLOCK,BLOCK_PER_KERNEL) 
__global__ void kernRemoveInvalidMarks(uint8_t* mark, size_t pitch, uint8_t* sim, size_t sim_pitch, size_t patterns, GPUNODE* node, uint32_t* fans, int start, int robust_f) {
	int tid = (blockIdx.y * blockDim.x) + threadIdx.x;
	int gid = (blockIdx.x) + start;
	const GPUNODE gate = node[gid];
	__shared__ int fots[128];
	if (threadIdx.x < gate.nfo) { fots[threadIdx.x] = FIN(fans,gate.offset,threadIdx.x+gate.nfi); }
	__syncthreads();
	if ((tid*4) < patterns) {
		// there must be at least one fan-out of current gate that is marked with our gate as a on-path gate, if not a po
		coalesce_t result;
		result.packed = (gate.po > 0) * 0x01010101;;
		for (uint16_t j = 0; j < gate.nfo; j++) {
			const GPUNODE fanout = node[fots[j]];
			if (robust_f == 1) {
				result.packed = result.packed | robust(gid,sim,sim_pitch,tid,fans, fanout).packed;
			} else {
				result.packed = result.packed | nonrobust(gid,sim,sim_pitch,tid,fans, fanout).packed;
			}
		}
		result.packed = result.packed & REF2D((coalesce_t*)mark,pitch,tid,gid).packed;
		REF2D((coalesce_t*)mark,pitch,tid,gid) = result;
	}
}
__global__ void kernRemoveInvalidMarks(uint8_t* mark, size_t pitch, uint8_t* sim, size_t sim_pitch, size_t patterns, GPUNODE* node, uint32_t* fans, int start) {
	int tid = (blockIdx.y * blockDim.x) + threadIdx.x;
	int gid = (blockIdx.x) + start;
	const GPUNODE gate = node[gid];
	__shared__ int fots[128];
	if (threadIdx.x < gate.nfo) { fots[threadIdx.x] = FIN(fans,gate.offset,threadIdx.x+gate.nfi); }
	__syncthreads();
	if ((tid*4) < patterns) {
		// there must be at least one fan-out of current gate that is marked with our gate as a on-path gate, if not a po
		coalesce_t result;
		result.packed = (gate.po > 0) * 0x01010101;;
		for (uint16_t j = 0; j < gate.nfo; j++) {
			if (gate.nfo < 128) {
				result.packed = (result.packed | REF2D((coalesce_t*)mark, pitch, tid, fots[j]).packed);
			} else {
				result.packed = (result.packed | REF2D((coalesce_t*)mark, pitch, tid, FIN(fans,gate.offset,j+gate.nfi)).packed);
			}
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
			kernMarkPathSegments<<<numBlocks,MARK_BLOCK>>>(input.gpu(chunk).data, input.gpu(chunk).pitch, results.gpu(chunk).data, results.gpu(chunk).pitch, results.gpu(chunk).width,ckt.gpu_graph(), ckt.offset(),  startGate, startPattern, robust_flag);
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
			kernRemoveInvalidMarks<<<numBlocks,INV_MARK_BLOCK>>>(results.gpu(chunk).data, results.gpu(chunk).pitch, input.gpu(chunk).data, input.gpu(chunk).pitch, results.gpu(chunk).width,ckt.gpu_graph(), ckt.offset(),  startGate);//, robust_flag);
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

