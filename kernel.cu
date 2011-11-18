#include <cuda.h>
#include "iscas.h"
#include "kernel.h"
#define N 32
#define PATTERNS 2

__global__ void NAND_gate(int i, int* fans, GPUNODE* graph, int *res, size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 0;
	int *row;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*width*sizeof(int));
		row[fans[graph[i].offset+graph[i].nfi]] = row[fans[graph[i].offset]];
		while (j < graph[i].nfi) {
			row[fans[graph[i].offset+graph[i].nfi]] = !(row[fans[graph[i].offset]] && row[fans[graph[i].offset+j]]);
			j++;
		}
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void FROM_gate(int i, int* fans,GPUNODE* graph, int *res, size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x;
	int *row;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*width*sizeof(int)); // get the current row?
		row[fans[graph[i].offset+graph[i].nfi]] = row[fans[graph[i].offset]];
		tid += blockDim.x * gridDim.x;
	}
}

void runGpuSimulation(int* results, size_t width, GPUNODE* ggraph, GPUNODE* graph, int maxid, LINE* line, int maxline, int* fan) {
	printf("Pre-simulation device memory check:\n");
	int *lvalues = (int*)malloc(sizeof(int)*width), *row;
	for (int r = 0;r < PATTERNS; r++) {
		lvalues = (int*)malloc(sizeof(int)*width);
		row = (int*)((char*)results + r*width*sizeof(int)); // get the current row?
		cudaMemcpy(lvalues,row,width*sizeof(int),cudaMemcpyDeviceToHost);
		for (int i = 0; i <width; i++) {
			printf("%d,%d:\t%d\n", r, i, lvalues[i]);
		}
	}

	for (int i = 0; i < maxid; i++) {
		printf("ID: %d, Type: %d\t", i, graph[i].type);
		switch (graph[i].type) {
			case NAND:
				printf("NAND Gate \n");
				NAND_gate<<<1,2>>>(i, fan, ggraph, results, width);
				break;
			case FROM:
				printf("FROM Gate \n");
				FROM_gate<<<1,2>>>(i, fan, ggraph, results, width);
				break;
			default:
				printf("Other Gate\n");
				break;
		}
		cudaThreadSynchronize();
	}
	printf("Post-simulation device memory check:\n");
	for (int r = 0;r < PATTERNS; r++) {
		lvalues = (int*)malloc(sizeof(int)*width);
		row = (int*)((char*)results + r*width*sizeof(int)); // get the current row?
		cudaMemcpy(lvalues,row,width*sizeof(int),cudaMemcpyDeviceToHost);
		for (int i = 0; i < width; i++) {
			printf("%d,%d:\t%d\n", r, i, lvalues[i]);
		}
	}
}
