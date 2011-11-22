#include <cuda.h>
#include <cassert>
#include "iscas.h"
#include "kernel.h"
#include "defines.h"

texture<int, 2> and2LUT;
texture<int, 2> nand2LUT;
texture<int, 2> or2LUT;
texture<int, 2> nor2LUT;
texture<int, 2> xor2LUT;
texture<int, 2> xnor2LUT;
texture<int, 2> stableLUT;
__global__ void XOR_gate(int i, int* fans, GPUNODE* graph, int *res, int PATTERNS,size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int val;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[graph[i].offset]];
		while (j < graph[i].nfi) {
			val = tex2D(xor2LUT, val, row[fans[graph[i].offset+j]]);
			j++;
		}
		row[fans[graph[i].offset+graph[i].nfi]] = val;
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}
__global__ void XNOR_gate(int i, int* fans, GPUNODE* graph, int *res, int PATTERNS,size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int val;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[graph[i].offset]];
		while (j < graph[i].nfi) {
			val = tex2D(xnor2LUT, val, row[fans[graph[i].offset+j]]);
			j++;
		}
		row[fans[graph[i].offset+graph[i].nfi]] = val;
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}
__global__ void OR_gate(int i, int* fans, GPUNODE* graph, int *res, int PATTERNS,size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int val;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[graph[i].offset]];
		while (j < graph[i].nfi) {
			val = tex2D(or2LUT, val, row[fans[graph[i].offset+j]]);
			j++;
		}
		row[fans[graph[i].offset+graph[i].nfi]] = val;
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}
__global__ void NOR_gate(int i, int* fans, GPUNODE* graph, int *res, int PATTERNS,size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int val;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[graph[i].offset]];
		while (j < graph[i].nfi) {
			val = tex2D(nor2LUT, val, row[fans[graph[i].offset+j]]);
			j++;
		}
		row[fans[graph[i].offset+graph[i].nfi]] = val;
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}
__global__ void AND_gate(int i, int* fans, GPUNODE* graph, int *res, int PATTERNS,size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int val;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[graph[i].offset]];
		while (j < graph[i].nfi) {
			val = tex2D(and2LUT, val, row[fans[graph[i].offset+j]]);
			j++;
		}
		row[fans[graph[i].offset+graph[i].nfi]] = val;
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}
__global__ void NAND_gate(int i, int* fans, GPUNODE* graph, int *res, int PATTERNS, size_t width , int pass, int* g) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int val;
	if (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[graph[i].offset]];
		g[tid] = fans[graph[i].offset];
		while (j < graph[i].nfi) {
			val = tex2D(nand2LUT, val, row[fans[graph[i].offset+j]]);
			j++;
		}
		if (pass == 1) {
			row[fans[graph[i].offset+graph[i].nfi]] = val;
		} else {
			row[fans[graph[i].offset+graph[i].nfi]] = tex2D(stableLUT, row[fans[graph[i].offset+graph[i].nfi]], val);  
		}
		__syncthreads();
	}
}

__global__ void FROM_gate(int i, int* fans,GPUNODE* graph, int *res, int PATTERNS, size_t width, int pass, int* g) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x;
	int *row;
	if (tid < PATTERNS) {
		g[tid] = fans[graph[i].offset+graph[i].nfi];
		row = (int*)((char*)res + tid*width*sizeof(int)); // get the current row?
		row[fans[graph[i].offset+graph[i].nfi]] = row[fans[graph[i].offset]];
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}
__global__ void INPT_gate(int i, int pi, ARRAY2D<int> results, ARRAY2D<int> input, GPUNODE* graph, int* fans,int pass, int* g) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, val;
	int *row;
	if (tid < results.height) {
		row = (int*)((char*)results.data + tid*results.width*sizeof(int)); // get the current row?
		val = *(input.data+(pi+input.width*tid));
		g[tid] = fans[graph[i].offset+graph[i].nfi];
		row[fans[graph[i].offset+graph[i].nfi]] = val;
	}
	printf("Hello thread %d, i=%d, input count: %d/%d input value=%d\n", threadIdx.x, i,pi+1,input.width, input.data[pi]) ;
	__syncthreads();
}



void loadLookupTables() {
	// Creating a set of static arrays that represent our LUTs
	int nand2[16] = {1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0};
	int and2[16]  = {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1};
	int nor2[16]  = {1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0};
	int or2[16]   = {0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
	int xnor2[16] = {1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1};
	int xor2[16]  = {0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0};
	int stable[4] = {S0, T0, T1, S1};
	// device memory arrays, required. 
	cudaArray *cuNandArray, *cuAndArray,*cuNorArray, *cuOrArray,*cuXnorArray,*cuXorArray, *cuStableArray;
	// generic formatting information. All of our arrays are the same, so sharing it shouldn't be a problem.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(int)*8,0,0,0,cudaChannelFormatKindUnsigned);
	
	// Allocating memory on the device.
	cudaMallocArray(&cuNandArray, &channelDesc, 4,4);
	cudaMallocArray(&cuAndArray, &channelDesc, 4,4);
	cudaMallocArray(&cuNorArray, &channelDesc, 4,4);
	cudaMallocArray(&cuOrArray, &channelDesc, 4,4);
	cudaMallocArray(&cuXnorArray, &channelDesc, 4,4);
	cudaMallocArray(&cuXorArray, &channelDesc, 4,4);
	cudaMallocArray(&cuStableArray, &channelDesc, 2,2);

	// Copying the LUTs Host->Device
	cudaMemcpyToArray(cuNandArray, 0,0, nand2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuAndArray, 0,0, and2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuNorArray, 0,0, nor2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuOrArray, 0,0, or2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXnorArray, 0,0, xnor2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXorArray, 0,0, xor2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuStableArray, 0,0, stable, sizeof(int)*4,cudaMemcpyHostToDevice);

	// Marking them as textures. LUTs should be in texture memory and cached on
	// access.
	cudaBindTextureToArray(and2LUT,cuAndArray,channelDesc);
	cudaBindTextureToArray(nand2LUT,cuNandArray,channelDesc);
	cudaBindTextureToArray(or2LUT,cuOrArray,channelDesc);
	cudaBindTextureToArray(nor2LUT,cuNorArray,channelDesc);
	cudaBindTextureToArray(xor2LUT,cuXorArray,channelDesc);
	cudaBindTextureToArray(xnor2LUT,cuXnorArray,channelDesc);
	cudaBindTextureToArray(stableLUT,cuStableArray,channelDesc);
}
void runGpuSimulation(ARRAY2D<int> results, ARRAY2D<int> inputs, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, ARRAY2D<LINE> line, int* fan, int pass = 1) {
	// Allocate a buffer memory for printf statements
	int *g;
	int piNumber = 0, curPI = 0;
	cudaMalloc((void**)&g,results.bwidth());
	cudaMemset(g, -1, sizeof(int)*results.height);
	for (int i = 0; i <= dgraph.width; i++) {
		DPRINT("ID: %d\tFanin: %d\tFanout: %d\tType: %d\t", i, graph[i].nfi, graph[i].nfo,graph[i].type);
		curPI = piNumber;
		switch (graph[i].type) {
			case 0:
				continue;
			case INPT:
				DPRINT("INPT Gate");
				INPT_gate<<<1,results.height>>>(i, curPI, results, inputs, dgraph.data, fan, 1, g);
				piNumber++;
				break;
			case XNOR:
				DPRINT("XNOR Gate");
				XNOR_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width);
			case XOR:
				DPRINT("XOR Gate");
				XOR_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width);
			case NOR:
				DPRINT("NOR Gate");
				NOR_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width);
			case OR:
				DPRINT("OR Gate");
				OR_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width);
			case AND:
				DPRINT("AND Gate");
				AND_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width);
			case NAND:
				DPRINT("NAND Gate");
				NAND_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width, 1,g);
				break;
			case FROM:
				DPRINT("FROM Gate");
				FROM_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width, 1,g);
				break;
			default:
				DPRINT("Other Gate");
				break;
		}
		DPRINT("\n");
	#ifndef NDEBUG
	// Routine to copy contents of our results array into host memory and print
	// it row-by-row.

	DPRINT("Post-simulation inspection results:\n");
	int *gres = (int*)malloc(sizeof(int)*results.height);
	cudaMemcpy(gres,g,results.height*sizeof(int),cudaMemcpyDeviceToHost);
	for (int r = 0;r < results.height; r++) {
		DPRINT("Gate %d: %d,%d\n", i, r, gres[r]);
	}
	free(gres);
#endif 	
	cudaThreadSynchronize();
	}

#ifndef NDEBUG
	// Routine to copy contents of our results array into host memory and print
	// it row-by-row.

	DPRINT("Post-simulation device results:\n");
	int *lvalues, *row;
	for (int r = 0;r < results.height; r++) {
		lvalues = (int*)malloc(results.bwidth());
		row = (int*)((char*)results.data + r*results.bwidth()); // get the current row?
		cudaMemcpy(lvalues,row,results.bwidth(),cudaMemcpyDeviceToHost);
		for (int i = 0; i < results.width; i++) {
			DPRINT("%d,%d:\t%d\n", r, i, lvalues[i]);
		}
		free(lvalues);
	}
#endif 
}
