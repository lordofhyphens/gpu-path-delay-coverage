#include <cuda.h>
#include <cassert>
#include "defines.h"
#include "iscas.h"
#include "simkernel.h"

texture<int, 2> and2LUT;
texture<int, 2> nand2LUT;
texture<int, 2> or2LUT;
texture<int, 2> nor2LUT;
texture<int, 2> xor2LUT;
texture<int, 2> xnor2LUT;
texture<int, 2> stableLUT;
texture<int, 1> notLUT;
__global__ void INPT_gate(int i, int pi, ARRAY2D<int> results, ARRAY2D<int> input, GPUNODE* graph, int* fans,int pass) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, val;
	int *row;
	if (tid < results.height) {
		row = (int*)((char*)results.data + tid*results.width*sizeof(int)); // get the current row?
		val = *(input.data+(pi+input.width*tid));
		if (pass > 1) {
			row[fans[graph[i].offset+graph[i].nfi]] = tex2D(stableLUT, row[fans[graph[i].offset+graph[i].nfi]], val);  
		} else {
			row[fans[graph[i].offset+graph[i].nfi]] = val;
		}
#ifdef GDEBUG // turn on GPU debugging printf statements.
		printf("Hello thread %d, i=%d, input count: %d/%d input value=%d\n", threadIdx.x, i,pi+1,input.width, input.data[pi]) ;
#endif
	}
	__syncthreads();
}

__global__ void LOGIC_gate(int i, GPUNODE* node, int* fans, int* res, size_t height, size_t width , int pass) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int goffset,nfi;
	int val;
	if (tid < height) {
		goffset = node[i].offset;
		nfi = node[i].nfi;
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		if (node[i].type != NOT) {
			val = row[fans[goffset]];
		} else {
			val = tex1D(notLUT, row[fans[goffset]]);
		}
		while (j < nfi) {
			switch(node[i].type) {
				case XOR:
					val = tex2D(xor2LUT, val, row[fans[goffset+j]]);
				case XNOR:
					val = tex2D(xnor2LUT, val, row[fans[goffset+j]]);
				case OR:
					val = tex2D(or2LUT, val, row[fans[goffset+j]]);
				case NOR:
					val = tex2D(nor2LUT, val, row[fans[goffset+j]]);
				case AND:
					val = tex2D(and2LUT, val, row[fans[goffset+j]]);
				case NAND:
					val = tex2D(nand2LUT, val, row[fans[goffset+j]]);
				case NOT:
					val = tex2D(nand2LUT, val, row[fans[goffset+j]]);

			}
			j++;
		}
		if (pass > 1 && node[i].type != FROM && node[i].type != BUFF) {
			row[fans[goffset+nfi]] = tex2D(stableLUT, row[fans[goffset+nfi]], val);  
		} else {
			row[fans[goffset+nfi]] = val;
		}
	}
	__syncthreads();
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
	int piNumber = 0, curPI = 0;
#ifndef NTIMING
	float elapsed;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
#endif // NTIMING
	for (int i = 0; i <= dgraph.width; i++) {
		DPRINT("ID: %d\tFanin: %d\tFanout: %d\tType: %d\t", i, graph[i].nfi, graph[i].nfo,graph[i].type);
		curPI = piNumber;
		switch (graph[i].type) {
			case 0:
				continue;
			case INPT:
				DPRINT("INPT Gate");
				INPT_gate<<<1,results.height>>>(i, curPI, results, inputs, dgraph.data, fan, pass);
				piNumber++;
				break;
			default:
				DPRINT("Logic Gate");
				LOGIC_gate<<<1,results.height>>>(i, dgraph.data, fan, results.data, results.height, results.width, pass);
				break;
		}
		DPRINT("\n");
		cudaDeviceSynchronize();
	}
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
