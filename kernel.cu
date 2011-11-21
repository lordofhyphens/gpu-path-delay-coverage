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

__global__ void XOR_gate(int i, int* fans, GPUNODE* graph, int *res, size_t width) {
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
__global__ void XNOR_gate(int i, int* fans, GPUNODE* graph, int *res, size_t width) {
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
__global__ void OR_gate(int i, int* fans, GPUNODE* graph, int *res, size_t width) {
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
__global__ void NOR_gate(int i, int* fans, GPUNODE* graph, int *res, size_t width) {
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
__global__ void AND_gate(int i, int* fans, GPUNODE* graph, int *res, size_t width) {
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
__global__ void NAND_gate(int i, int* fans, GPUNODE* graph, int *res, size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int val;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[graph[i].offset]];
		while (j < graph[i].nfi) {
			val = tex2D(nand2LUT, val, row[fans[graph[i].offset+j]]);
			j++;
		}
		row[fans[graph[i].offset+graph[i].nfi]] = val;
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}

__global__ void FROM_gate(int i, int* fans,GPUNODE* graph, int *res, size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x;
	int *row;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*width*sizeof(int)); // get the current row?
		row[fans[graph[i].offset+graph[i].nfi]] = row[fans[graph[i].offset]];
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}

void loadLookupTables() {
	// Creating a set of static arrays that represent our LUTs
	int nand2[4] = {1, 1, 1, 0};
	int and2[4] = {0, 0, 0, 1};
	int nor2[4] = {1, 0, 0, 0};
	int or2[4] = {0,1,1,1};
	int xnor2[4] = {1,0,0,1};
	int xor2[4] = {0,1,1,0};
	// device memory arrays, required. 
	cudaArray* cuNandArray, *cuAndArray,*cuNorArray, *cuOrArray,*cuXnorArray,*cuXorArray;
	// generic formatting information. All of our arrays are the same, so sharing it shouldn't be a problem.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(int)*8,0,0,0,cudaChannelFormatKindUnsigned);
	
	// Allocating memory on the device.
	cudaMallocArray(&cuNandArray, &channelDesc, 2,2);
	cudaMallocArray(&cuAndArray, &channelDesc, 2,2);
	cudaMallocArray(&cuNorArray, &channelDesc, 2,2);
	cudaMallocArray(&cuOrArray, &channelDesc, 2,2);
	cudaMallocArray(&cuXnorArray, &channelDesc, 2,2);
	cudaMallocArray(&cuXorArray, &channelDesc, 2,2);

	// Copying the LUTs Host->Device
	cudaMemcpyToArray(cuNandArray, 0,0, nand2, sizeof(int)*4,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuAndArray, 0,0, and2, sizeof(int)*4,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuNorArray, 0,0, nor2, sizeof(int)*4,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuOrArray, 0,0, or2, sizeof(int)*4,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXnorArray, 0,0, xnor2, sizeof(int)*4,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXorArray, 0,0, xor2, sizeof(int)*4,cudaMemcpyHostToDevice);

	// Marking them as textures. LUTs should be in texture memory and cached on
	// access.
	cudaBindTextureToArray(and2LUT,cuAndArray,channelDesc);
	cudaBindTextureToArray(nand2LUT,cuNandArray,channelDesc);
	cudaBindTextureToArray(or2LUT,cuOrArray,channelDesc);
	cudaBindTextureToArray(nor2LUT,cuNorArray,channelDesc);
	cudaBindTextureToArray(xor2LUT,cuXorArray,channelDesc);
	cudaBindTextureToArray(xnor2LUT,cuXnorArray,channelDesc);
}
void runGpuSimulation(int* results, size_t width, GPUNODE* ggraph, GPUNODE* graph, int maxid, LINE* line, int maxline, int* fan) {

	for (int i = 0; i <= maxid; i++) {
		printf("ID: %d\tFanin: %d\tFanout: %d\tType: %d\t", i, graph[i].nfi, graph[i].nfo,graph[i].type);
		switch (graph[i].type) {
			case 0:
				continue;
			case XNOR:
				printf("XNOR Gate");
				XNOR_gate<<<1,PATTERNS>>>(i, fan, ggraph, results, width);
			case XOR:
				printf("XOR Gate");
				XOR_gate<<<1,PATTERNS>>>(i, fan, ggraph, results, width);
			case NOR:
				printf("NOR Gate");
				NOR_gate<<<1,PATTERNS>>>(i, fan, ggraph, results, width);
			case OR:
				printf("OR Gate");
				OR_gate<<<1,PATTERNS>>>(i, fan, ggraph, results, width);
			case AND:
				printf("AND Gate");
				AND_gate<<<1,PATTERNS>>>(i, fan, ggraph, results, width);
			case NAND:
				printf("NAND Gate");
				NAND_gate<<<1,PATTERNS>>>(i, fan, ggraph, results, width);
				break;
			case FROM:
				printf("FROM Gate");
				FROM_gate<<<1,PATTERNS>>>(i, fan, ggraph, results, width);
				break;
			default:
				printf("Other Gate");
				break;
		}
		printf("\n");
		cudaThreadSynchronize();
	}

	// Routine to copy contents of our results array into host memory and print
	// it row-by-row.

	printf("Post-simulation device results:\n");
	int *lvalues = (int*)malloc(sizeof(int)*width), *row;
	for (int r = 0;r < PATTERNS; r++) {
		lvalues = (int*)malloc(sizeof(int)*width);
		row = (int*)((char*)results + r*width*sizeof(int)); // get the current row?
		cudaMemcpy(lvalues,row,width*sizeof(int),cudaMemcpyDeviceToHost);
		for (int i = 0; i < width; i++) {
			printf("%d,%d:\t%d\n", r, i, lvalues[i]);
		}
		free(lvalues);
	}
}
