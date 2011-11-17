#include <cuda.h>
#include "iscas.h"
#define N 32
__global__ void OR_gate(int i, GPUNODE* graph, int type, int **res) {
	int tid = blockIdx.x;
	int cnt = 0;
	int tmp;
	while (tid < N) {
		tmp = res[graph[i].fin[cnt].prev][tid]; 
		while (cnt < graph[i].nfi){
			res[i][tid] = (tmp || res[graph[i].fin[cnt].prev][tid]);
			cnt++;
		}
	}
}
__global__ void AND_gate(int i, GPUNODE* graph, int type, int **res) {
	int tid = blockIdx.x;
	int cnt = 0;
	int tmp;
	while (tid < N) {
		tmp = res[graph[i].fin[cnt].prev][tid]; 
		while (cnt < graph[i].nfi) {
			res[i][tid] = (tmp && res[graph[i].fin[cnt].prev][tid]);
			cnt++;
		}
	}
}
