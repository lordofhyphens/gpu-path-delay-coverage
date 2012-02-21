#include <cuda.h>
#include "simkernel.h"

void HandleSimError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleSimError( err, __FILE__, __LINE__ ))
texture<int, 2> and2LUT;
texture<int, 2> nand2LUT;
texture<int, 2> or2LUT;
texture<int, 2> nor2LUT;
texture<int, 2> xor2LUT;
texture<int, 2> xnor2LUT;
texture<int, 2> stableLUT;
texture<int, 1> notLUT;

__device__ char simLUT(int type, char val, char r) {
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
__global__ void kernSimulateP1(GPUNODE* graph, char* pi_data, size_t pi_pitch, size_t pi_offset, char* output_data, size_t pitch, size_t pattern_count, int* fanout_index, int start_offset, int startPattern) {
	int tid = (blockIdx.y * SIM_BLOCK) + threadIdx.x;
	int gid = blockIdx.x+start_offset;
	int pid = tid + startPattern;
	char rowcache;
	char *row, r, val;
	int goffset, nfi, j,type;
	if (pid < pattern_count)  {
		row = ((char*)output_data + gid*pitch)+tid; // get the line row for the current gate
		goffset = graph[gid].offset;
		nfi = graph[gid].nfi;
		type = graph[gid].type;
		__syncthreads();
		rowcache = REF2D(char, output_data, pitch, tid, FIN(fanout_index, goffset, 0));
		switch (type) {
			case INPT: val = REF2D(char, pi_data, pi_pitch, pid+pi_offset, gid); break;
			case FROM: val = REF2D(char, output_data, pitch, tid, FIN(fanout_index, goffset, 0)); break;
			default: 
					// we're guaranteed at least one fanin per 
					// gate if not on an input.
					__syncthreads();
					if (type != NOT) { val = rowcache; } else { val = !(BIN(rowcache)); }
					j = 1;
					while (j < nfi) {
						__syncthreads();
						r = REF2D(char, output_data, pitch, tid, FIN(fanout_index, goffset, j));  
						val = simLUT(type,val,r);
						j++;
					}
		}
		*row = val;
	}
}
__global__ void kernSimulateP2(GPUNODE* graph, char* pi_data, size_t pi_pitch, size_t pi_offset, char* output_data, size_t pitch,size_t pattern_count,  int* fanout_index, int start_offset, int startPattern) {
	int tid = (blockIdx.y * SIM_BLOCK) + threadIdx.x, prev=0;
	int gid = blockIdx.x+start_offset;
	int pid = tid + startPattern;
	char rowcache;
	char *row, r;
	int goffset, nfi, val, j,type;

	if (pid < pattern_count) {
		row = ((char*)output_data + gid*pitch)+tid; // get the line row for the current gate
		goffset = graph[gid].offset;
		nfi = graph[gid].nfi;
		type = graph[gid].type;
		prev = *row;

		rowcache = ((char*)output_data+(fanout_index[goffset]*pitch))[tid];
		switch (type) {
			case INPT: val = BIN(REF2D(char, pi_data, pi_pitch, (pid+pi_offset), gid)); break;
			case FROM: val = BIN(REF2D(char, output_data, pitch, tid, FIN(fanout_index, goffset, 0))); break;
			default: 
					// we're guaranteed at least one fanin per 
					// gate if not on an input.
					__syncthreads();
					if (type != NOT) { val = BIN(rowcache); } else { val = !(BIN(rowcache)); }
					j = 1;
					while (j < nfi) {
						__syncthreads();
						r = REF2D(char, output_data, pitch, tid, FIN(fanout_index,goffset,j));
						val = simLUT(type,val,BIN(r));
						j++;
					}
		}
		*row = tex2D(stableLUT,prev,val);
	}
}

void loadSimLUTs() {
	int nand2[16] = {1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0};
	int and2[16]  = {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1};
	int nor2[16]  = {1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0};
	int or2[16]   = {0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
	int xnor2[16] = {1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1};
	int xor2[16]  = {0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0};
	int stable[4] = {S0, T0, T1, S1};
	int not_gate[4] = {1, 1, 0, 0};

	// device memory arrays, required. 
	cudaArray *cuNandArray, *cuAndArray,*cuNorArray, *cuOrArray,*cuXnorArray,*cuXorArray, *cuNotArray,*cuStableArray;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();

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
	HANDLE_ERROR(cudaMemcpyToArray(cuNandArray, 0,0, nand2, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndArray, 0,0, and2, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuNorArray, 0,0, nor2, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrArray, 0,0, or2, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXnorArray, 0,0, xnor2, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorArray, 0,0, xor2, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuStableArray, 0,0, stable, sizeof(int)*4,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuNotArray, 0,0, not_gate, sizeof(int)*4,cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaBindTextureToArray(and2LUT,cuAndArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(nand2LUT,cuNandArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(or2LUT,cuOrArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(nor2LUT,cuNorArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xor2LUT,cuXorArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xnor2LUT,cuXnorArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(stableLUT,cuStableArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(notLUT,cuNotArray,channelDesc));
}

float gpuRunSimulation(GPU_Data& results, GPU_Data& inputs, GPU_Circuit& ckt, int pass = 1) {
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
	loadSimLUTs(); // set up our lookup tables for simulation.
	int startGate;
//	DPRINT("Results block width: %lu\n",results.block_width());
	int blockcount_y = (int)(results.gpu().width/SIM_BLOCK) + ((results.gpu().width%SIM_BLOCK) > 0);
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	size_t startPattern = 0;
	for (unsigned int chunk = 0; chunk < results.size(); chunk++) {
		startGate = 0;
		DPRINT("Patterns to process in block %u: %lu\n", chunk, results.gpu(chunk).width);
		for (int i = 0; i <= ckt.levels(); i++) {
			int levelsize = ckt.levelsize(i);
			do { 
				int simblocks = min(MAX_BLOCKS, levelsize);
				dim3 numBlocks(simblocks,blockcount_y);
//				DPRINT("%s:%d - %d %d %d %d \n", __FILE__,__LINE__, i, simblocks, blockcount_y, levelsize);
				if (pass > 1) {
					kernSimulateP2<<<numBlocks,SIM_BLOCK>>>(ckt.gpu_graph(), inputs.gpu().data, inputs.gpu().pitch,
							chunk*results.gpu(chunk).width, results.gpu(chunk).data, results.gpu().pitch, 
							inputs.block_width(), ckt.offset(), startGate, startPattern);
				} else {
					kernSimulateP1<<<numBlocks,SIM_BLOCK>>>(ckt.gpu_graph(), inputs.gpu().data, inputs.gpu().pitch, 
							chunk*results.gpu(chunk).width, results.gpu(chunk).data, results.gpu().pitch, 
							inputs.block_width(), ckt.offset(), startGate, startPattern);
				}
				startGate += simblocks;
				if (levelsize > MAX_BLOCKS) {
					levelsize -= MAX_BLOCKS;
				} else {
					levelsize = 0;
				}
			} while (levelsize > 0); 
			cudaDeviceSynchronize();
			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
		}
		startPattern += results.gpu(chunk).width;
	}
	// We're done simulating at this point.
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}
void debugSimulationOutput(ARRAY2D<char> results, std::string outfile = "simdebug.log") {
#ifndef NDEBUG
	char *lvalues;
	std::ofstream ofile(outfile.c_str());
	lvalues = (char*)malloc(results.height*results.pitch);
	cudaMemcpy2D(lvalues,results.pitch,results.data,results.pitch,results.width,results.height,cudaMemcpyDeviceToHost);
	for (unsigned int r = 0;r < results.width; r++) {
		ofile << "Vector " << r << ":\t";
		for (unsigned int i = 0; i < results.height; i++) {
			char z = REF2D(char, lvalues, results.pitch, r, i);
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
