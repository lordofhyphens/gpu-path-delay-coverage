#include <cuda.h>
#include "iscas.h"
#include "kernel.h"
#define N 32
#define PATTERNS 2

__global__ void AND_gate(int i, int* fans, LINE* line, GPUNODE graph, int *res, size_t pitch) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x;
	while (tid < PATTERNS) {
		int* row = (int*)((char*)res + tid*pitch);
		int j = 0;
		while (j < graph.nfi) {
			j++;
			row[fans[graph.offset+graph.nfi]] = (row[fans[graph.offset+graph.nfi]] && row[fans[graph.offset+j]]);
		}
	}
}

__global__ void OR_gate(int i, GPUNODE* graph, LINE* lgraph, int* offsets, int type, int **res) {
	int tid = blockIdx.x;
	int cnt = 0;
	int tmp;
	while (tid < N) {
		tmp = res[lgraph[offsets[graph[i].offset+cnt]].prev][tid]; 
		while (cnt < graph[i].nfi){
			res[i][tid] = (tmp || res[lgraph[offsets[graph[i].offset+cnt]].prev][tid]);
			cnt++;
		}
		cnt = 0;
	}
}
__global__ void NAND_gate(int i, int* fans, LINE* line, GPUNODE graph, int *res, size_t pitch) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x;
	while (tid < PATTERNS) {
		int* row = (int*)((char*)res + tid*pitch);
		int j = 0;
		while (j < graph.nfi) {
			j++;
			row[fans[graph.offset+graph.nfi]] = !(row[fans[graph.offset+graph.nfi]] && row[fans[graph.offset+j]]);
		}
		printf("TID %d:\t%d\n",tid, row[fans[graph.offset+graph.nfi]]);
	}
}

__global__ void FROM_gate(int i, int* fans, LINE* line, GPUNODE graph, int *res, size_t pitch) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x;
	while (tid < PATTERNS) {
		int* row = (int*)((char*)res + tid*pitch);
		row[fans[graph.offset+graph.nfi]] = row[fans[graph.offset]];
		printf("TID %d:\t%d\n",tid, row[fans[graph.offset+graph.nfi]]);
	}
}

void runGpuSimulation(int* results, size_t width, GPUNODE* graph, int maxid, LINE* line, int maxline, int* fan) {
	printf("Pre-simulation device memory check:\n");
	int *lvalues1 = (int*)malloc(sizeof(int)*width*2);
	cudaMemcpy(lvalues1,results,width*2*sizeof(int),cudaMemcpyDeviceToHost);
	for (int i = 0; i < 2*width; i++) {
		printf("%d:\t%d\n", i, lvalues1[i]);
	}

	for (int i = 0; i < maxid; i++) {
		switch (graph[i].type) {
			case AND:
				AND_gate<<<1,2>>>(i, fan, line, graph[i], results, width);
				break;
			case OR:
				break;
			case NAND:
				NAND_gate<<<1,2>>>(i, fan, line, graph[i], results, width);
				break;
			case FROM:
				FROM_gate<<<1,2>>>(i, fan, line, graph[i], results, width);
				break;
			default:
				break;
		}
	}
	printf("Post-simulation device memory check:\n");
	lvalues1 = (int*)malloc(sizeof(int)*width*2);
	cudaMemcpy(lvalues1,results,width*2*sizeof(int),cudaMemcpyDeviceToHost);
	for (int i = 0; i < 2*width; i++) {
		printf("%d:\t%d\n", i, lvalues1[i]);
	}
}
