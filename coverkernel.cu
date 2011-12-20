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

// reference: design book 1, page 38.
__global__ void kernCountCoverage(int toffset, int *results, int *history, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x, nfi, goffset;
	int *row, *historyRow;
	int *current, *historyCount;
	__shared__ char rowids[50]; // handle up to fanins of 1000 /
	if (tid < height) {
		row = (int*)((char*)results + tid*(width)*sizeof(int));
		if (tid == 0) {
			historyRow = (int*)malloc(sizeof(int)*width);
			memset(historyRow, 0, sizeof(int)*width);
		} else {
			historyRow = (int*)((char*)history + (tid-1)*(width)*sizeof(int));
		}
		current = (int*)malloc(sizeof(int)*width);
		historyCount = (int*)malloc(sizeof(int)*width);
		for (int i = 0; i < ncount; i++) {
			current[i] = 0;
			historyCount[i] = 0;
		}
		for (int i = ncount; i >= 0; i--) {
			nfi = node[i].nfi;
			if (tid == 0) {
				goffset = node[i].offset;
				// preload all of the fanin line #s for this gate to shared memory.
				// Guaranteed 1 cycle access time.
				for (int j = 0; j < nfi;j++) {
					rowids[j] = (char)fans[goffset+j];
				}
			}
			__syncthreads();
			if (node[i].po) {
				current[i] = (row[i] > historyRow[i]); // only set = 1 if there's a new line here
				historyCount[i] = historyRow[i];
			}
			switch(node[i].type) {
				case 0: continue;
				case FROM:
						// Add the current fanout count to the fanin if this line is marked (and the history isn't).
						current[rowids[0]] += current[i]*(row[rowids[0]] > historyRow[rowids[0]]);
						historyCount[rowids[0]] += historyCount[i]*(historyRow[rowids[0]]);
						break;
				case INPT:
						continue;
				default: 
						for (int fin = 0; fin < node[i].nfi; fin++) {
							// if the fanout total is 0 but this line is marked (and the history isn't), add a path to the count.
							// If the fanout total is > 1 and this line is marked (and the history isn't), assign the fanout total to the fanins.
							historyCount[rowids[fin]] += (historyRow[rowids[fin]] || historyCount[i] > 1) * historyCount[i];
							current[rowids[fin]] += ((row[rowids[fin]] > historyRow[rowids[fin]]) || current[i] > 1) * current[i] + historyCount[i]*(current[i] == 0 && row[rowids[fin]] > historyRow[rowids[fin]]);
						}

			}
		}
		for (int i = 0; i < ncount; i++) {
			row[i] = current[i];
		}
		free(current);
		free(historyCount);
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
	DPRINT("Path Count results\n");
	DPRINT("Line:   \t");
	for (int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (int r = 0;r < results.height; r++) {
		lvalues = (int*)malloc(results.bwidth());
		row = (int*)((char*)results.data + r*results.bwidth()); // get the current row?
		cudaMemcpy(lvalues,row,results.bwidth(),cudaMemcpyDeviceToHost);
		
		DPRINT("%s %3d:\t", "Vector",r);
		for (int i = 0; i < results.width; i++) {
			DPRINT("%3d ", lvalues[i] == 0 ? -1:lvalues[i]);
		}
		DPRINT("\n");
		free(lvalues);
	}
#endif 
}
float gpuCountPaths(ARRAY2D<int> results, ARRAY2D<int> history, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan) {
#ifndef NTIMING
	float elapsed = 0.0, cover = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
#endif
	kernCountCoverage<<<1,results.height>>>(0, results.data, history.data,dgraph.data, fan, results.width, results.height, dgraph.width);
	cudaDeviceSynchronize();
#ifndef NTIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed,start,stop);
	DPRINT("Cover time: %2f \n", elapsed);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
#endif
	kernSumAll<<<1,1>>>(0, results.data, history.data,dgraph.data, fan, results.width, results.height, dgraph.width);
#ifndef NTIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsed+cover;
#else
	return 0.0;
#endif
}
int returnPathCount(ARRAY2D<int> results) {
	int tmp;
	cudaMemcpy(&tmp, results.data, sizeof(int), cudaMemcpyDeviceToHost);
	return tmp;
}
