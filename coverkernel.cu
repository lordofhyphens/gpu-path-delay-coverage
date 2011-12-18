#include <cuda.h>
#include "defines.h"
#include "coverkernel.h"

// badly sums everything and places it into row[0][0]
__global__ void kernSumAll(int toffset, int *results, int *history, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, nfi, goffset;
	int *row;
	__shared__ int sum;
	if (tid < 1) {
		sum = 0;
		for (int j = 0; j < height; j++) {
			row = (int*)((char*)results + j*(width)*sizeof(int));
			for (int c = ncount; c >= 0; c--) {
				goffset = node[c].offset;
				nfi = node[c].nfi;
				if (node[c].type == INPT)
					sum = sum + row[fans[goffset+nfi]];
				//printf("Sum Count: %d\n",sum);
			}
		}
		row = (int*)((char*)results);
		row[0] = sum;
	}
}
__global__ void kernCountCoverage(int toffset, int *results, int *history, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x, nfi, goffset;
	int *row, *historyRow;
	int *current;
	__shared__ char rowids[1000]; // handle up to fanins of 1000 /
	if (tid < height) {
		row = (int*)((char*)results + tid*(width)*sizeof(int));
		if (tid == 0) {
			historyRow = (int*)malloc(sizeof(int)*width);
			memset(historyRow, 0, sizeof(int)*width);
		} else {
			historyRow = (int*)((char*)history + (tid-1)*(width)*sizeof(int));
		}
		current = (int*)malloc(sizeof(int)*width);
		for (int i = 0; i < ncount; i++) {
			current[i] = 0;
		}
		for (int i = ncount; i >= 0; i--) {
			nfi = node[i].nfi;
			if (tid == 0) {
				goffset = node[i].offset;
				// preload all of the fanin line #s for this gate to shared memory.
				for (int j = 0; j < nfi;j++) {
					rowids[j] = (char)fans[goffset+j];
				}
			}
			__syncthreads();
			if (node[i].po) {
				current[i] = (row[i] > historyRow[i]); // only set = 1 if there's a new line here
			}
			switch(node[i].type) {
				case 0: continue;
				case FROM:
//						printf("T: %d G %d Fanout: %d, fanin: %d/%d\n",tid, i, current[i], row[rowids[0]],historyRow[rowids[0]]);
						current[rowids[0]] += current[i]*(row[rowids[0]] > historyRow[rowids[0]]);
						break;
				case INPT:
						continue;
				default: 
						for (int fin = 0; fin < node[i].nfi; fin++) {
//							printf("T: %d G %d Fanout: %d, fanin %d: %d/%d\n",tid, i, current[i], rowids[fin], row[rowids[fin]],historyRow[rowids[fin]]);
							current[rowids[fin]] += ((row[rowids[fin]] > historyRow[rowids[fin]]) || current[i] > 1) * current[i] + (current[i] == 0 && row[rowids[fin]] > historyRow[rowids[fin]]);
						}

			}
		}
		for (int i = 0; i < ncount; i++) {
			row[i] = current[i];
		}
		free(current);
		if (tid == 0) {
			free(historyRow);
		}
	}

}
void debugCoverOutput(ARRAY2D<int> results) {
#ifndef NDEBUG
	// Routine to copy contents of our results array into host memory and print
	// it row-by-row.
	int *lvalues, *row;
	DPRINT("Post-union results\n");
	DPRINT("Line:   \t");
	for (int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (int r = 0;r < results.height; r++) {
		lvalues = (int*)malloc(results.bwidth());
		row = (int*)((char*)results.data + r*results.bwidth()); // get the current row?
		cudaMemcpy(lvalues,row,results.bwidth(),cudaMemcpyDeviceToHost);
		
		DPRINT("%s %d:\t", "Vector",r);
		for (int i = 0; i < results.width; i++) {
			DPRINT("%2d ", lvalues[i]);
		}
		DPRINT("\n");
		free(lvalues);
	}
#endif 
}
float gpuCountPaths(ARRAY2D<int> results, ARRAY2D<int> history, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan) {
#ifndef NTIMING
	float elapsed = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
#endif
	kernCountCoverage<<<1,results.height>>>(0, results.data, history.data,dgraph.data, fan, results.width, results.height, dgraph.width);
	cudaDeviceSynchronize();
	kernSumAll<<<1,1>>>(0, results.data, history.data,dgraph.data, fan, results.width, results.height, dgraph.width);
#ifndef NTIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsed;
#else
	return 0.0;
#endif
}
int returnPathCount(ARRAY2D<int> results) {
	int tmp;
	cudaMemcpy(&tmp, results.data, sizeof(int), cudaMemcpyDeviceToHost);
	return tmp;
}
