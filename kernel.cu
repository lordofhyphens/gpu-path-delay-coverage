#include <cuda.h>
#include "iscas.h"
#include "kernel.h"
#define N 32
#define PATTERNS 2

__global__ void NAND_gate(int i, int* fans, GPUNODE* graph, int *res, size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x;
	int *row;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*width*sizeof(int));
		row[fans[graph[i].offset+graph[i].nfi]] = row[fans[graph[i].offset]];
	}
}

__global__ void FROM_gate(int i, int* fans,GPUNODE* graph, int *res, size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x;
	int *row;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*width*sizeof(int)); // get the current row?
		row[fans[graph[i].offset+graph[i].nfi]] = row[fans[graph[i].offset]];
	}
}

void runGpuSimulation(int* results, size_t width, GPUNODE* ggraph, GPUNODE* graph, int maxid, LINE* line, int maxline, int* fan) {
	printf("Pre-simulation device memory check:\n");
	int *lvalues1 = (int*)malloc(sizeof(int)*width*2);
	cudaMemcpy(lvalues1,results,width*2*sizeof(int),cudaMemcpyDeviceToHost);
	for (int i = 0; i < 2*width; i++) {
		printf("%d:\t%d\n", i, lvalues1[i]);
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
	lvalues1 = (int*)malloc(sizeof(int)*width*2);
	cudaMemcpy(lvalues1,results,width*2*sizeof(int),cudaMemcpyDeviceToHost);
	for (int i = 0; i < 2*width; i++) {
		printf("%d:\t%d\n", i, lvalues1[i]);
	}
}
