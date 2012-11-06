#include "markkernel.h"
#include <cuda.h>
#undef LOGEXEC
#undef MARK_BLOCK
#define MARK_BLOCK 128
#define BLOCK_PER_KERNEL 8
void HandleMarkError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleMarkError( err, __FILE__, __LINE__ ))
texture<uint8_t, 2> and2OutputPropLUT;
texture<uint8_t, 2> or2OutputPropLUT;
texture<uint8_t, 2> xor2OutputPropLUT;

void loadPropLUTs() {
	// Creating a set of static arrays that represent our LUTs
		// Addressing for the propagations:
	// 2 4x4 groups such that 
	uint8_t and2_output_prop[16]= {0,0,0,0,0,2,1,1,0,1,1,0,0,1,0,1};
	uint8_t or2_output_prop[16] = {2,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1};
	uint8_t xor2_output_prop[16]= {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};

	cudaExtent volumeSize = make_cudaExtent(4,4,2);
	// device memory arrays, required. 
	cudaArray *cuAndOutpProp, *cuOrOutpProp, *cuXorOutpProp;
	// generic formatting information. All of our arrays are the same, so sharing it shouldn't be a problem.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();
	// Allocating memory on the device.
		
	HANDLE_ERROR(cudaMallocArray(&cuAndOutpProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuOrOutpProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuXorOutpProp, &channelDesc, 4,4));

	// Copying the LUTs Host->Device
	HANDLE_ERROR(cudaMemcpyToArray(cuAndOutpProp, 0,0, and2_output_prop, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrOutpProp, 0,0, or2_output_prop, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorOutpProp, 0,0, xor2_output_prop, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	
	// Marking them as textures. LUTs should be in texture memory and cached on
	// access.
	HANDLE_ERROR(cudaBindTextureToArray(and2OutputPropLUT,cuAndOutpProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(or2OutputPropLUT,cuOrOutpProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xor2OutputPropLUT,cuXorOutpProp,channelDesc));
}
DEVICE uint8_t markeval_out (const uint8_t f1, const uint8_t f2, const int type) {
	switch(type) {
		case AND:
		case NAND:
			return tex2D(and2OutputPropLUT, f1, f2);
		case OR:
		case NOR:
			return tex2D(or2OutputPropLUT, f1, f2);
		case XOR:
		case XNOR:
			return tex2D(xor2OutputPropLUT, f1, f2);
	}
	return 0xff;
}
extern "C" __launch_bounds__(MARK_BLOCK,BLOCK_PER_KERNEL) 
__global__ void kernMarkPathSegments(uint8_t *sim, size_t sim_pitch, uint8_t* mark, size_t pitch, size_t patterns, GPUNODE* node, uint32_t* fans, int start, int startPattern) {
	int tid = (blockIdx.y * blockDim.x) + threadIdx.x;
	int gid = (blockIdx.x) + start;
	uint8_t resultCache = 0;
	uint8_t cache;
	int pass = 0;
	if (tid < patterns) {
		const GPUNODE& gate = node[gid];
		const uint8_t rowCache = (REF2D(uint8_t,sim,sim_pitch,tid,gid) > 1);
		assert(rowCache == 1 || rowCache == 0);
		
		// switching based on value causes divergence, switch based on node type instead.
		switch(gate.type) {
			case INPT:
				if (gate.nfo == 0 && gate.po < 1) {
					resultCache = 0; // on the odd case that an input is literally connected to nothing, this is not a path.
				} else {
					resultCache = rowCache;// Otherwise we can mark this.
				}
				break;
			case FROM: // For FROM, it's equal to its fan-in
			case BUFF:
			case DFF:
				// For inverter and buffer gates, mark if and only if a fan-in is marked.
			case NOT: resultCache = REF2D(uint8_t,mark,pitch,tid,FIN(fans,gate.offset,0)); break;
			case OR:  // For the normal gates, set the fan-out based on the fan-ins. 
			case NOR: // There's a LUT for each basic gate type.
			case XOR:
			case XNOR:
			case NAND:
			case AND:
				for (uint16_t fin1 = 0; fin1 < gate.nfi; fin1++) {
					pass = 0;
					for (uint16_t fin2 = 0; fin2 < gate.nfi; fin2++) {
						if (fin1 != fin2) {
							const uint8_t sim_fin1 = REF2D(uint8_t,sim,sim_pitch,tid,FIN(fans,gate.offset,fin1));
							const uint8_t sim_fin2 = REF2D(uint8_t,sim,sim_pitch,tid,FIN(fans,gate.offset,fin2));
							cache = markeval_out(sim_fin1, sim_fin2, gate.type);
							assert(cache < 3); // ensure no invalid responses from function
							pass += (cache > 1);
							if (fin1 == 0) {
								resultCache = (cache > 0)*(pass < gate.nfi);
							} else {
								resultCache = (cache > 0)*(pass < gate.nfi)*resultCache;
							}
						}
					}
				}
				break;
			default: break;
		}
		// stick the contents of resultCache into the mark array
		REF2D(uint8_t,mark,pitch,tid,gid) = resultCache*rowCache;
	}
}

float gpuMarkPaths(GPU_Data& results, GPU_Data& input, GPU_Circuit& ckt) {
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting before we hit this point.
	loadPropLUTs();
	int startGate;
	int blockcount_y;
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	int startPattern = 0;
	for (unsigned int chunk = 0; chunk < input.size(); chunk++) {
		blockcount_y = (int)(results.gpu(chunk).width/MARK_BLOCK) + ((results.gpu(chunk).width% MARK_BLOCK) > 0);
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
		startPattern += input.gpu(chunk).width;
		cudaDeviceSynchronize();
	}
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif // NTIMING
#ifdef LOGEXEC
	debugMarkOutput(&results, "gpumark.log");
#endif // LOGEXEC
#ifndef NTIMING
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}

void debugMarkOutput(GPU_Data* results, std::string outfile = "simdebug.log") {
#ifndef NDEBUG
	std::ofstream ofile(outfile.c_str());
	size_t t = 0;
	for (size_t chunk = 0; chunk < results->size(); chunk++) {
		uint8_t *lvalues;
		lvalues = (uint8_t*)malloc(results->gpu(chunk).height*results->gpu(chunk).pitch);
		cudaMemcpy2D(lvalues,results->gpu().pitch,results->gpu(chunk).data,results->gpu(chunk).pitch,results->gpu(chunk).width,results->gpu(chunk).height,cudaMemcpyDeviceToHost);
		for (unsigned int r = 0;r < results->gpu(chunk).width; r++) {
			ofile << "Vector " << t << ":\t";
			for (unsigned int i = 0; i < results->gpu(chunk).height; i++) {
				uint8_t z = REF2D(uint8_t, lvalues, results->gpu(chunk).pitch, r, i);
				switch(z) {
					case 0:
						ofile  << std::setw(OUTJUST) << "N" << " "; break;
					case 1:
						ofile  << std::setw(OUTJUST) << "Y" << " "; break;
					default:
						ofile << std::setw(OUTJUST) << (int)z << " "; break;
				}
			}
			ofile << std::endl;
			t++;
		}
		free(lvalues);
	}
	ofile.close();
#endif

}
void debugMarkOutput(ARRAY2D<uint8_t> results, std::string outfile) {
#ifndef NDEBUG
	uint8_t *lvalues;
	std::ofstream ofile(outfile.c_str());
//	ofile << "Line:   \t";
//	for (unsigned int i = 0; i < results.height; i++) {
//		ofile << std::setw(OUTJUST) << i << " ";
//	}
//	ofile << std::endl;
	lvalues = (uint8_t*)malloc(results.height*results.pitch);
	cudaMemcpy2D(lvalues,results.pitch,results.data,results.pitch,results.width,results.height,cudaMemcpyDeviceToHost);
	for (unsigned int r = 0;r < results.width; r++) {
		ofile << "Vector " << r << ":\t";
		for (unsigned int i = 0; i < results.height; i++) {
			uint8_t z = REF2D(uint8_t, lvalues, results.pitch, r, i);
			switch(z) {
				case 0:
					ofile  << std::setw(OUTJUST) << "N" << " "; break;
				case 1:
					ofile  << std::setw(OUTJUST) << "Y" << " "; break;
				default:
					ofile << std::setw(OUTJUST) << (int)z << " "; break;
			}
		}
		ofile << std::endl;
	}
	free(lvalues);
	ofile.close();
#endif
}

