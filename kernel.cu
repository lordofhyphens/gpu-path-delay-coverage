#include <cuda.h>
#include "iscas.h"
#define N 32


__global__ void IN_gate(int i, GPUNODE* graph, LINE* lgraph, int* offsets, int type, int** res) {

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
__global__ void AND_gate(int i, GPUNODE* graph, LINE* lgraph, int* offsets, int type, int **res) {
	int tid = blockIdx.x;
	int cnt = 0;
	int tmp;
	while (tid < N) {
		tmp = res[lgraph[offsets[graph[i].offset+cnt]].prev][tid]; 
		while (cnt < graph[i].nfi) {
			res[i][tid] = (tmp && res[lgraph[offsets[graph[i].offset+cnt]].prev][tid]);
			cnt++;
		}
	}
}
