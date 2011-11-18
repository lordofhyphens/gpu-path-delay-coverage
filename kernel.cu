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

__global__ void NAND_gate(int i, int* fans, GPUNODE* graph, int *res, size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		if (graph[i].nfi <= 2) {
			row[fans[graph[i].offset+graph[i].nfi]] = tex2D(nand2LUT, row[fans[graph[i].offset]], row[fans[graph[i].offset+1]]);
		} else {
			row[fans[graph[i].offset+graph[i].nfi]] = row[fans[graph[i].offset]];
			while (j < graph[i].nfi) {
				row[fans[graph[i].offset+graph[i].nfi]] = tex2D(nand2LUT, row[fans[graph[i].offset+graph[i].nfi]], row[fans[graph[i].offset+j]]);
				j++;
			}
		}
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
	int nand2[4] = {1, 1, 1, 0};
	int and2[4] = {0, 0, 0, 1};
	int nor2[4] = {1, 0, 0, 0};
	int or2[4] = {0,1,1,1};
	int xnor2[4] = {1,0,0,1};
	int xor2[4] = {0,1,1,0};
	cudaArray* cuNandArray, *cuAndArray,*cuNorArray, *cuOrArray,*cuXnorArray,*cuXorArray;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(int)*8,0,0,0,cudaChannelFormatKindUnsigned);
	cudaMallocArray(&cuNandArray, &channelDesc, 2,2);
	cudaMallocArray(&cuAndArray, &channelDesc, 2,2);
	cudaMallocArray(&cuNorArray, &channelDesc, 2,2);
	cudaMallocArray(&cuOrArray, &channelDesc, 2,2);
	cudaMallocArray(&cuXnorArray, &channelDesc, 2,2);
	cudaMallocArray(&cuXorArray, &channelDesc, 2,2);
	cudaMemcpyToArray(cuNandArray, 0,0, nand2, sizeof(int)*4,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuAndArray, 0,0, and2, sizeof(int)*4,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuNorArray, 0,0, nor2, sizeof(int)*4,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuOrArray, 0,0, or2, sizeof(int)*4,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXnorArray, 0,0, xnor2, sizeof(int)*4,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXorArray, 0,0, xor2, sizeof(int)*4,cudaMemcpyHostToDevice);

	cudaBindTextureToArray(and2LUT,cuAndArray,channelDesc);
	cudaBindTextureToArray(nand2LUT,cuNandArray,channelDesc);
	cudaBindTextureToArray(or2LUT,cuOrArray,channelDesc);
	cudaBindTextureToArray(nor2LUT,cuNorArray,channelDesc);
	cudaBindTextureToArray(xor2LUT,cuXorArray,channelDesc);
	cudaBindTextureToArray(xnor2LUT,cuXnorArray,channelDesc);
}
void runGpuSimulation(int* results, size_t width, GPUNODE* ggraph, GPUNODE* graph, int maxid, LINE* line, int maxline, int* fan) {
	int *lvalues = (int*)malloc(sizeof(int)*width), *row;
/*	printf("Pre-simulation device memory check:\n");
	for (int r = 0;r < PATTERNS; r++) {
		lvalues = (int*)malloc(sizeof(int)*width);
		row = (int*)((char*)results + r*width*sizeof(int)); // get the current row?
		cudaMemcpy(lvalues,row,width*sizeof(int),cudaMemcpyDeviceToHost);
		for (int i = 0; i <width; i++) {
			printf("%d,%d:\t%d\n", r, i, lvalues[i]);
		}
	}
	free(lvalues);
*/
	for (int i = 0; i <= maxid; i++) {
		printf("ID: %d\tFanin: %d\tFanout: %d\tType: %d\t", i, graph[i].nfi, graph[i].nfo,graph[i].type);
		switch (graph[i].type) {
			case 0:
				continue;
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
	printf("Post-simulation device results:\n");
	for (int r = 0;r < PATTERNS; r++) {
		lvalues = (int*)malloc(sizeof(int)*width);
		row = (int*)((char*)results + r*width*sizeof(int)); // get the current row?
		cudaMemcpy(lvalues,row,width*sizeof(int),cudaMemcpyDeviceToHost);
		for (int i = 0; i < width; i++) {
			printf("%d,%d:\t%d\n", r, i, lvalues[i]);
		}
		free(lvalues);
	}
/*
	lvalues = (int*)malloc(sizeof(int)*width*PATTERNS);
	cudaMemcpy(lvalues,results,PATTERNS*width*sizeof(int),cudaMemcpyDeviceToHost);
	for (int i = 0; i < width*PATTERNS; i++) {
		printf("%d:\t%d\n", i, lvalues[i]);
	}
	free(lvalues);
*/
}
