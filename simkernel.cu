#include <cuda.h>
#include <cassert>
#include "defines.h"
#include "iscas.h"
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
__global__ void kernSimulateP1(GPUNODE* graph, int* pi_data, size_t pi_pitch, char* output_data, size_t pitch,size_t pattern_count, int* fanout_index, int start_offset) {
	int tid = (blockIdx.y * blockDim.y) + threadIdx.x;
	int gid = blockIdx.x+start_offset;
	__shared__ char rowcache[SIM_BLOCK];
	char *row, r, val;
	int goffset, nfi, j,type;
	if (tid < pattern_count) {
		row = ((char*)output_data + gid*pitch); // get the line row for the current gate
		goffset = graph[gid].offset;
		nfi = graph[gid].nfi;
		type = graph[gid].type;

		__syncthreads();
		rowcache[threadIdx.x] = ((char*)output_data+(fanout_index[goffset]*pitch))[tid];
		switch (type) {
			case INPT:
				val = pi_data[gid+pi_pitch*(tid)];
				break;
			default: 
					// we're guaranteed at least one fanin per 
					// gate if not on an input.
					__syncthreads();
					if (type != NOT) {
						val = rowcache[threadIdx.x];
					} else {
						val = tex1D(notLUT, rowcache[threadIdx.x]);
					}

					j = 1;
					while (j < nfi) {
						__syncthreads();
						r = REF2D(char,output_data,pitch,FIN(fanout_index,goffset,j),tid); //((char*)output_data+(fanout_index[goffset+j]*pitch))[tid]; 
						switch(type) {
							case XOR:
								val = tex2D(xor2LUT, val, r);break;
							case XNOR:
								val = tex2D(xnor2LUT, val, r);break;
							case OR:
								val = tex2D(or2LUT, val, r);break;
							case NOR:
								val = tex2D(nor2LUT, val, r);break;
							case AND:
								val = tex2D(and2LUT, val, r);break;
							case NAND:
								val = tex2D(nand2LUT, val, r);break;
						}
						j++;
					}
		}
		row[tid] = val;
	}
}
__global__ void kernSimulateP2(GPUNODE* graph, int* pi_data, size_t pi_pitch, char* output_data, size_t pitch,size_t pattern_count,  int* fanout_index, int start_offset) {
	int tid = (blockIdx.y * blockDim.y) + threadIdx.x, prev=0;
	int gid = blockIdx.x+start_offset;
	__shared__ char rowcache[SIM_BLOCK];
	char *row, r;
	int goffset, nfi, val, j,type;

	if (tid < pattern_count) {
		row = ((char*)output_data + gid*pitch)+tid; // get the line row for the current gate
		goffset = graph[gid].offset;
		nfi = graph[gid].nfi;
		type = graph[gid].type;
		prev = *row;

		rowcache[threadIdx.x] = ((char*)output_data+(fanout_index[goffset]*pitch))[tid];
		switch (type) {
			case INPT:
				val = pi_data[gid+pi_pitch*(tid)];
				break;
			default: 
					// we're guaranteed at least one fanin per 
					// gate if not on an input.
					__syncthreads();
					if (type != NOT) {
						val = rowcache[threadIdx.x];
					} else {
						val = tex1D(notLUT, rowcache[threadIdx.x]);
					}

					j = 1;
					while (j < nfi) {
						__syncthreads();
						r = ((char*)output_data+(fanout_index[goffset+j]*pitch))[tid]; 
						switch(type) {
							case XOR:
								val = tex2D(xor2LUT, val, r );break;
							case XNOR:
								val = tex2D(xnor2LUT, val, r);break;
							case OR:
								val = tex2D(or2LUT, val, r);break;
							case NOR:
								val = tex2D(nor2LUT, val, r);break;
							case AND:
								val = tex2D(and2LUT, val, r);break;
							case NAND:
								val = tex2D(nand2LUT, val, r);break;
						}
						j++;
					}
		}
		if (type == FROM || type == BUFF)
			*row = val;
		else {
			*row = tex2D(stableLUT,prev,val);
		}
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

float gpuRunSimulation(ARRAY2D<char> results, ARRAY2D<int> inputs, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan, int maxlevels, int pass = 1) {
	loadSimLUTs(); // set up our lookup tables for simulation.
	int startGate = 0;
	int *gatesinLevel;
	gatesinLevel = new int[maxlevels];
	for (int i = 0; i < maxlevels; i++) {
		gatesinLevel[i] = 0;
		for (unsigned int j = 0; j < results.width; j++) {
			if (graph[j].level == i) {
				gatesinLevel[i]++;
			}
		}
	}
	int blockcount_y = (int)(results.height/SIM_BLOCK) + (results.height%SIM_BLOCK > 0);
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	for (int i = 0; i < maxlevels; i++) {
		dim3 numBlocks(gatesinLevel[i],blockcount_y);
		if (pass > 1) {
			kernSimulateP2<<<numBlocks,SIM_BLOCK>>>(dgraph.data,inputs.data, inputs.width, results.data, results.pitch, inputs.height, fan, startGate);
		} else {
			kernSimulateP1<<<numBlocks,SIM_BLOCK>>>(dgraph.data,inputs.data, inputs.width, results.data, results.pitch, inputs.height, fan, startGate);
		}
		startGate += gatesinLevel[i];
		cudaDeviceSynchronize();
	}
	free(gatesinLevel);
	// We're done simulating at this point.
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}

void debugSimulationOutput(ARRAY2D<char> results, int pass = 1) {
#ifndef NDEBUG
	char *lvalues, *row;
	DPRINT("Post-simulation device results, pass %d:\n\n", pass);
	DPRINT("Vector:   \t");
	for (unsigned int i = 0; i < results.height; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (unsigned int r = 0;r < results.width; r++) {
		lvalues = (char*)malloc(results.height*sizeof(char));
		row = ((char*)results.data + r*results.pitch); // get the current row?
		cudaMemcpy(lvalues,row,results.height*sizeof(char),cudaMemcpyDeviceToHost);
		DPRINT("%s %d:\t", pass > 1 ? "Line " : "Line ",r);
		for (unsigned int i = 0; i < results.height; i++) {
			switch(lvalues[i]) {
				case S0:
					DPRINT("S0 "); break;
				case S1:
					DPRINT("S1 "); break;
				case T0:
					DPRINT("T0 "); break;
				case T1:
					DPRINT("T1 "); break;
				default:
					DPRINT("%2d ", lvalues[i]); break;
			}
		}
		DPRINT("\n");
		free(lvalues);
	}
#endif
}
