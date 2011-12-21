#include <cuda.h>
#include <cassert>
#include "defines.h"
#include "iscas.h"
#include "simkernel.h"

#define THREAD_PER_BLOCK 256
texture<int, 2> and2LUT;
texture<int, 2> nand2LUT;
texture<int, 2> or2LUT;
texture<int, 2> nor2LUT;
texture<int, 2> xor2LUT;
texture<int, 2> xnor2LUT;
texture<int, 2> stableLUT;
texture<int, 1> notLUT;

__global__ void kernSimulate(GPUNODE* graph, int* res, int* input, int* fans, size_t iwidth, size_t width, size_t height, int pass) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	__shared__ int rowids[100]; // handle up to fanins of 1000 / 
	int pi = 0;
	int *row;
	int goffset, nfi, val, j,type, r;
	if (tid < height) {
		row = (int*)((char*)res + tid*width*sizeof(int)); // get the current row?
		for (int i = 0; i < width; i++) {
			nfi = graph[i].nfi;
			if (threadIdx.x == 0) { // first thread in every block does the preload.
				goffset = graph[i].offset;
//				printf("Offset (gate %d): %d\n", i, goffset);
				// preload all of the fanin line #s for this gate to shared memory.
				for (int j = 0; j < nfi;j++) {
					rowids[j] = fans[goffset+j];
//					printf("Gate %d, fanin %d = %d (wrote %d)\n",i, j, fans[goffset+j],rowids[j]);
				}
					
			}
			__syncthreads();
			type = graph[i].type;
			switch (type) {
				case 0: break;
				case INPT:
						val = input[(pi+iwidth*tid)];
						if (pass > 1) {
							row[i] = tex2D(stableLUT, row[i], val);  
						} else {
							row[i] = val;
						}
						pi++;
						break;
				default: 
						// we're guaranteed at least one fanin per 
						// gate if not on an input.
						__syncthreads();
						if (rowids[0] < 0) {
							printf("T: %d Node %d, Type %d, Rowid0 %d\n", tid, i, graph[i].type, rowids[0]);
						}
						if (graph[i].type != NOT) {
							val = row[rowids[0]];
						} else {
							val = tex1D(notLUT, row[rowids[0]]);
						}
						
						j = 1;
						while (j < nfi) {
							r = row[rowids[j]]; 
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
									if (tid == 664) { 
//										printf("\n");
									}
									val = tex2D(nand2LUT, val, r);break;
							}
							j++;
						}
						if (pass > 1 && type != FROM && type != BUFF) {
							row[i] = tex2D(stableLUT, row[i], val);  
						} else {
							row[i] = val;
						}
			}

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
	cudaMallocArray(&cuNandArray, &channelDesc, 4,4);
	cudaMallocArray(&cuAndArray, &channelDesc, 4,4);
	cudaMallocArray(&cuNorArray, &channelDesc, 4,4);
	cudaMallocArray(&cuOrArray, &channelDesc, 4,4);
	cudaMallocArray(&cuXnorArray, &channelDesc, 4,4);
	cudaMallocArray(&cuXorArray, &channelDesc, 4,4);
	cudaMallocArray(&cuStableArray, &channelDesc, 2,2);
	cudaMallocArray(&cuNotArray, &channelDesc, 4,1);

	// Copying the static arrays given to device memory.
	cudaMemcpyToArray(cuNandArray, 0,0, nand2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuAndArray, 0,0, and2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuNorArray, 0,0, nor2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuOrArray, 0,0, or2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXnorArray, 0,0, xnor2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXorArray, 0,0, xor2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuStableArray, 0,0, stable, sizeof(int)*4,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuNotArray, 0,0, not_gate, sizeof(int)*4,cudaMemcpyHostToDevice);

	cudaBindTextureToArray(and2LUT,cuAndArray,channelDesc);
	cudaBindTextureToArray(nand2LUT,cuNandArray,channelDesc);
	cudaBindTextureToArray(or2LUT,cuOrArray,channelDesc);
	cudaBindTextureToArray(nor2LUT,cuNorArray,channelDesc);
	cudaBindTextureToArray(xor2LUT,cuXorArray,channelDesc);
	cudaBindTextureToArray(xnor2LUT,cuXnorArray,channelDesc);
	cudaBindTextureToArray(stableLUT,cuStableArray,channelDesc);
	cudaBindTextureToArray(notLUT,cuNotArray,channelDesc);
}

float gpuRunSimulation(ARRAY2D<int> results, ARRAY2D<int> inputs, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan, int pass = 1) {
	loadSimLUTs(); // set up our lookup tables for simulation.
#ifndef NTIMING
	float elapsed;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
#endif // NTIMING
	int blockcount = (int)(results.height/THREAD_PER_BLOCK) + (results.height%THREAD_PER_BLOCK > 0);
//	DPRINT("Block count: %d, threads: %d\n", blockcount, THREAD_PER_BLOCK);
	kernSimulate<<<blockcount,THREAD_PER_BLOCK>>>(dgraph.data,results.data, inputs.data,fan,inputs.width, results.width, results.height, pass);
	cudaDeviceSynchronize();

	// We're done simulating at this point.
#ifndef NTIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}

void debugSimulationOutput(ARRAY2D<int> results, int pass = 1) {
#ifndef NDEBUG
	int *lvalues, *row;
	DPRINT("Post-simulation device results, pass %d:\n\n", pass);
	DPRINT("Line:   \t");
	for (int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (int r = 0;r < results.height; r++) {
		lvalues = (int*)malloc(results.bwidth());
		row = (int*)((char*)results.data + r*results.bwidth()); // get the current row?
		cudaMemcpy(lvalues,row,results.bwidth(),cudaMemcpyDeviceToHost);
		DPRINT("%s %d:\t", pass > 1 ? "Vector" : "Pattern",r);
		for (int i = 0; i < results.width; i++) {
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
