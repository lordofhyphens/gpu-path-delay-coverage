#include <cuda.h>
#include "simkernel.h"
#undef LOGEXEC
#define SIM_BLOCK 768
void HandleSimError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleSimError( err, __FILE__, __LINE__ ))
texture<uint8_t, 2> and2LUT;
texture<uint8_t, 2> nand2LUT;
texture<uint8_t, 2> or2LUT;
texture<uint8_t, 2> nor2LUT;
texture<uint8_t, 2> xor2LUT;
texture<uint8_t, 2> xnor2LUT;
texture<uint8_t, 2> stableLUT;
texture<uint8_t, 1> notLUT;

__device__ uint8_t simLUT(int type, uint8_t val, uint8_t r) {
	switch(type) {
		case XOR: return tex2D(xor2LUT, val, r );
		case XNOR: return tex2D(xnor2LUT, val, r);
		case OR: return tex2D(or2LUT, val, r);
		case NOR: return tex2D(nor2LUT, val, r);
		case AND: return tex2D(and2LUT, val, r);
		case NAND: return tex2D(nand2LUT, val, r);
		default:
			return 0;
	}
}
__global__ void kernSimulateP1(GPUNODE* graph, uint8_t* pi_data, size_t pi_pitch, size_t pi_offset, uint8_t* output_data, size_t pitch, size_t max_patterns, size_t pattern_count, uint32_t* fanout_index, int start_offset, int startPattern) {
	int tid = (blockIdx.y * SIM_BLOCK) + threadIdx.x;
	int gid = blockIdx.x+start_offset;
	int pid = tid + startPattern;
	int pid2 = (tid+startPattern == pattern_count -1 ? 0: tid+startPattern+1);
	uint8_t *row, r, val;
	int goffset, nfi, j,type;
	if (tid < max_patterns && pid < pattern_count )  {
		row = ((uint8_t*)output_data + gid*pitch)+tid; // get the line row for the current gate
		goffset = graph[gid].offset;
		nfi = graph[gid].nfi;
		type = graph[gid].type;
		__syncthreads();
		switch (type) {
			case INPT: val = tex2D(stableLUT,REF2D(uint8_t, pi_data, pi_pitch, pid+pi_offset, gid),REF2D(uint8_t, pi_data, pi_pitch, pid2+pi_offset, gid)); break;
			case DFF:
			case BUFF:
			case FROM: val = REF2D(uint8_t, output_data, pitch, tid, FIN(fanout_index, goffset, 0)); break;
			default: 
					// we're guaranteed at least one fanin per 
					// gate if not on an input.
					__syncthreads();
					if (type != NOT) { val = REF2D(uint8_t, output_data, pitch, tid, FIN(fanout_index, goffset, 0)); } 
					else { val = tex1D(notLUT, REF2D(uint8_t, output_data, pitch, tid, FIN(fanout_index, goffset, 0))); }
					j = 1;
					while (j < nfi) {
						__syncthreads();
						r = REF2D(uint8_t, output_data, pitch, tid, FIN(fanout_index, goffset, j));  
						val = simLUT(type,val,r);
						j++;
					}
		}
		if (val > T1) {
			assert(val <= T1);
		}
		*row = val;
	}
}

void loadSimLUTs() {
	uint8_t nand2[16] = {S1, S1, S1, S1, S1, S0, T1, T0, S1, T1, T1, S1, S1, T0, S1, T0};
	uint8_t and2[16]  = {S0, S0, S0, S0, S0, S1, T0, T1, S0, T0, T0, S0, S0, T1, S0, T1};
	uint8_t nor2[16]  = {S1, S0, T1, T0, S0, S0, S0, S0, T1, S0, T1, S0, T0, S0, S0, T0};
	uint8_t or2[16]   = {S0, S1, T0, T1, S1, S1, S1, S1, T0, S1, T0, S1, T1, S1, S1, T1};
	uint8_t xnor2[16] = {S1, S0, T1, T0, S0, S1, T0, T1, T1, T0, S1, S0, T0, T1, S0, S1};
	uint8_t xor2[16]  = {S0, S1, T0, T1, S1, S0, T1, T0, T0, T1, S0, S1, T1, T0, S1, S0};
	uint8_t stable[4] = {S0, T0, T1, S1};
	uint8_t not_gate[4] = {S1, S0, T1, T0};

	// device memory arrays, required. 
	cudaArray *cuNandArray, *cuAndArray,*cuNorArray, *cuOrArray,*cuXnorArray,*cuXorArray, *cuNotArray,*cuStableArray;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();

	// Allocate space in device memory for the LUTs. 
	HANDLE_ERROR(cudaMallocArray(&cuNandArray, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuAndArray, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuNorArray, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuOrArray, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuXnorArray, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuXorArray, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuStableArray, &channelDesc, 2,2));
	HANDLE_ERROR(cudaMallocArray(&cuNotArray, &channelDesc, 4,1));

	// Copying the static arrays given to device memory.
	HANDLE_ERROR(cudaMemcpyToArray(cuNandArray, 0,0, nand2, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndArray, 0,0, and2, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuNorArray, 0,0, nor2, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrArray, 0,0, or2, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXnorArray, 0,0, xnor2, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorArray, 0,0, xor2, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuStableArray, 0,0, stable, sizeof(uint8_t)*4,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuNotArray, 0,0, not_gate, sizeof(uint8_t)*4,cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaBindTextureToArray(and2LUT,cuAndArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(nand2LUT,cuNandArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(or2LUT,cuOrArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(nor2LUT,cuNorArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xor2LUT,cuXorArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xnor2LUT,cuXnorArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(stableLUT,cuStableArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(notLUT,cuNotArray,channelDesc));
}

float gpuRunSimulation(GPU_Data& results, GPU_Data& inputs, GPU_Circuit& ckt, uint8_t pass = 1) {
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
	loadSimLUTs(); // set up our lookup tables for simulation.
	int startGate;
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	size_t startPattern = 0;
	for (unsigned int chunk = 0; chunk < results.size(); chunk++) {
		const int blockcount_y = (int)(results.gpu(chunk).width/SIM_BLOCK) + ((results.gpu(chunk).width%SIM_BLOCK) > 0);
		startGate = 0;
		//DPRINT("Patterns to process in block %u: %lu\n", chunk, results.gpu(chunk).width);
		for (uint32_t i = 0; i <= ckt.levels(); i++) {
			int levelsize = ckt.levelsize(i);
			do { 
				int simblocks = min(MAX_BLOCKS, levelsize);
				dim3 numBlocks(simblocks,blockcount_y);
			//	DPRINT("Working on %lu patterns, %d gates in level %d\n",  results.gpu(chunk).width, simblocks, i);

				kernSimulateP1<<<numBlocks,SIM_BLOCK>>>(ckt.gpu_graph(), inputs.gpu().data, inputs.gpu().pitch,
						0, results.gpu(chunk).data, results.gpu(chunk).pitch, 
						results.gpu(chunk).width, inputs.block_width(), ckt.offset(), startGate, startPattern);

				startGate += simblocks;
				if (levelsize > MAX_BLOCKS) {
					levelsize -= MAX_BLOCKS;
				} else {
					levelsize = 0;
				}
			} while (levelsize > 0); 
			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
		}
		startPattern += results.gpu(chunk).width;
		cudaDeviceSynchronize();
	}
	// We're done simulating at this point.
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif // NTIMING
#ifdef LOGEXEC
		debugSimulationOutput(&results, "gpusim-p2.log");
#endif //LOGEXEC
#ifndef NTIMING
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}
void debugSimulationOutput(GPU_Data* results, std::string outfile = "simdebug.log") {
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
				case S0:
					ofile  << std::setw(OUTJUST+1) << "S0 "; break;
				case S1:
					ofile  << std::setw(OUTJUST+1) << "S1 "; break;
				case T0:
					ofile  << std::setw(OUTJUST+1) << "T0 "; break;
				case T1:
					ofile  << std::setw(OUTJUST+1) << "T1 "; break;
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
void debugSimulationOutput(ARRAY2D<uint8_t> results, std::string outfile = "simdebug.log") {
#ifndef NDEBUG
	uint8_t *lvalues;
	std::ofstream ofile(outfile.c_str());
	lvalues = (uint8_t*)malloc(results.height*results.pitch);
	cudaMemcpy2D(lvalues,results.pitch,results.data,results.pitch,results.width,results.height,cudaMemcpyDeviceToHost);
	for (unsigned int r = 0;r < results.width; r++) {
		ofile << "Vector " << r << ":\t";
		for (unsigned int i = 0; i < results.height; i++) {
			uint8_t z = REF2D(uint8_t, lvalues, results.pitch, r, i);
			switch(z) {
				case S0:
					ofile  << std::setw(OUTJUST+1) << "S0 "; break;
				case S1:
					ofile  << std::setw(OUTJUST+1) << "S1 "; break;
				case T0:
					ofile  << std::setw(OUTJUST+1) << "T0 "; break;
				case T1:
					ofile  << std::setw(OUTJUST+1) << "T1 "; break;
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
