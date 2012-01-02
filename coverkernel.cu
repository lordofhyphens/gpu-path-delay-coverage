#include <cuda.h>
#include "defines.h"
#include "coverkernel.h"

#define THREAD_PER_BLOCK 256
// badly sums everything and places it into row[0][0]
__global__ void kernSumAll(int toffset, char *results, char *history, GPUNODE* node, int* fans, size_t width, size_t height, size_t pitch, int ncount) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, nfi, goffset;
	char *row;
	__shared__ int sum;
	if (tid < 1) {
		sum = 0;
		for (int j = 0; j < height; j++) {
			row = (char*)((char*)results + j*(pitch));
			for (int c = ncount-1; c >= 0; c--) {
				goffset = node[c].offset;
				nfi = node[c].nfi;
				if (node[c].type == INPT)
					sum = sum + row[fans[goffset+nfi]];
				//printf("Sum Count: %d\n",sum);
			}
		}
		row = ((char*)results);
		row[0] = sum;
	}
}

// reference: design book 1, page 38.
__global__ void kernCountCoverage(int toffset, char *results, char *history, GPUNODE* node, int* fans, size_t width, size_t height, size_t pitch, size_t hpitch, int ncount) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x, nfi, goffset;
	char *row, *historyRow;
	int *current, *historyCount;
	__shared__ int rowids[50]; // handle up to fanins of 50 /
	if (tid < height) {
		row = ((char*)results + tid*pitch);
		if (tid == 0) {
			historyRow = (char*)malloc(sizeof(char)*width);
			memset(historyRow, 0, sizeof(char)*width);
		} else {
			historyRow = ((char*)history + (tid-1)*hpitch);
		}
		current = (int*)malloc(sizeof(int)*width);
		historyCount = (int*)malloc(sizeof(int)*width);
		for (int i = 0; i < ncount; i++) {
			current[i] = 0;
			historyCount[i] = 0;
		}
		for (int i = ncount-1; i >= 0; i--) {
			nfi = node[i].nfi;
			if (tid == 0) {
				goffset = node[i].offset;
				// preload all of the fanin line #s for this gate to shared memory.
				// Guaranteed 1 cycle access time.
				for (int j = 0; j < nfi;j++) {
					rowids[j] = fans[goffset+j];
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
void debugCoverOutput(ARRAY2D<char> results) {
#ifndef NDEBUG
	// Routine to copy contents of our results array into host memory and print
	// it row-by-row.
	char *lvalues, *row;
	DPRINT("Path Count results\n");
	DPRINT("Line:   \t");
	for (unsigned int i = 0; i < results.width; i++) {
		DPRINT("%3d ", i);
	}
	DPRINT("\n");
	for (unsigned int r = 0;r < results.height; r++) {
		lvalues = (char*)malloc(results.pitch);
		row = ((char*)results.data + r*results.bwidth()); // get the current row?
		cudaMemcpy(lvalues,row,results.pitch,cudaMemcpyDeviceToHost);
		
		DPRINT("%s %3d:\t", "Vector",r);
		for (unsigned int i = 0; i < results.width; i++) {
			DPRINT("%3d ", lvalues[i] == 0 ? 255:lvalues[i]);
		}
		DPRINT("\n");
		free(lvalues);
	}
#endif 
}
float gpuCountPaths(ARRAY2D<char> results, ARRAY2D<char> history, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan) {
#ifndef NTIMING
	float elapsed = 0.0;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif
	int blockcount = (int)(results.height/THREAD_PER_BLOCK) + (results.height%THREAD_PER_BLOCK > 0);
	kernCountCoverage<<<blockcount,THREAD_PER_BLOCK>>>(0, results.data, history.data,dgraph.data, fan, results.width, results.height, results.pitch, history.pitch, dgraph.width);
	cudaDeviceSynchronize();
#ifndef NTIMING
#endif
//	kernSumAll<<<1,1>>>(0, results.data, history.data,dgraph.data, fan, results.width, results.height, results.pitch,dgraph.width);
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start,stop));
	return elapsed;
#else
	return 0.0;
#endif
}
char returnPathCount(ARRAY2D<char> results) {
	char tmp;
	cudaMemcpy(&tmp, results.data, sizeof(char), cudaMemcpyDeviceToHost);
	return tmp;
}
