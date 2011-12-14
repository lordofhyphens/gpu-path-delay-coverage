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
	int tid = blockIdx.x * gridDim.x + threadIdx.x, nfi, goffset, vecnum = toffset + tid, val, tempr, temph;
	int *row, *rowHistory;
	int *rowResults,*historyResults;
	__shared__ int h_cache[2048];
	__shared__ int r_cache[2048];
	// For every node, count paths through this node
	if (tid < height) {
		rowResults = (int*)malloc(sizeof(int)*width);
		historyResults = (int*)malloc(sizeof(int)*width);
		row = (int*)((char*)results + vecnum*(width)*sizeof(int));
		rowHistory = (int*)((char*)history + (vecnum-1)*(width)*sizeof(int));
		for (int i = 0; i < width; i++) {
			rowResults[i] = 0; 
			historyResults[i] = 0;
		}
		if (vecnum == 0) {
			// handle the initial vector separately, just count paths.
			for (int c = ncount; c >= 0; c--) {
				nfi = node[c].nfi;
				goffset = node[c].offset;
				if (node[c].po) {
					rowResults[fans[goffset+nfi]] = row[fans[goffset+nfi]]*1;
				}
				switch (node[c].type) {
					case 0: break;
					case INPT: break;
					case FROM:
						rowResults[fans[goffset]] += row[fans[goffset+nfi]]*(rowResults[fans[goffset]]);
					break;
					default:
						for (int i = 0; i < nfi;i++)
							rowResults[fans[goffset+i]] = rowResults[fans[goffset+nfi]];

				}
			}
			for (int i = 0; i < width; i++) {
				row[i] = rowResults[i];
			}

		} else {
			for (int c = ncount; c >= 0; c--) {
				nfi = node[c].nfi;
				goffset = node[c].offset;
				if (node[c].po) {
					tempr = 0;
					temph = rowHistory[fans[goffset+nfi]]*1;
					val = row[fans[goffset+nfi]] > rowHistory[fans[goffset+nfi]];
					r_cache[threadIdx.x*2] = tempr;
					r_cache[(threadIdx.x*2)+1] = tempr + temph;
					h_cache[threadIdx.x*2] = temph;
					h_cache[(threadIdx.x*2)+1] = 0;
					rowResults[fans[goffset+nfi]] = r_cache[(threadIdx.x*2)+val];
					historyResults[fans[goffset+nfi]] = h_cache[(threadIdx.x*2)+val];
				}
				switch (node[c].type) {
					case 0: continue;
					case INPT: break;
					case FROM:
							   tempr = rowResults[fans[goffset]];
							   tempr += row[fans[goffset+nfi]]*(rowResults[fans[goffset+nfi]]);
							   temph = historyResults[fans[goffset]];
							   temph += rowHistory[fans[goffset+nfi]]*(historyResults[fans[goffset+nfi]]);
							   val = row[fans[goffset]] > rowHistory[fans[goffset]];
							   r_cache[threadIdx.x*2] = tempr;
							   r_cache[threadIdx.x*2+1] = tempr + temph;
							   h_cache[threadIdx.x*2] = temph;
							   h_cache[threadIdx.x*2+1] = 0;
							   rowResults[fans[goffset]] = r_cache[(threadIdx.x*2)+val];
							   historyResults[fans[goffset]] = h_cache[(threadIdx.x*2)+val];
							   break;
					default:
							   for (int i = 0; i < nfi;i++) {
								   rowResults[fans[goffset+i]] = rowResults[fans[goffset+nfi]];
								   historyResults[fans[goffset+i]] = historyResults[fans[goffset+nfi]];
								   val = row[fans[goffset+i]] > rowHistory[fans[goffset+i]];
								   r_cache[threadIdx.x*2]     = rowResults[fans[goffset+i]];
								   r_cache[(threadIdx.x*2)+1] = rowResults[fans[goffset+i]] + historyResults[fans[goffset+i]];
								   h_cache[threadIdx.x*2]     = historyResults[fans[goffset+i]];
								   h_cache[(threadIdx.x*2)+1] = 0;
								   rowResults[fans[goffset+i]] = r_cache[(threadIdx.x*2)+val];
								   historyResults[fans[goffset+i]] = h_cache[(threadIdx.x*2)+val];
							   }
				} 
			}
			for (int i = 0; i < width; i++) {
				row[i] = rowResults[i];
			}
		}

		free(rowResults);
		free(historyResults);
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
	DPRINT("rw %d, rh %d, gw %d, gh %d\n", results.width, results.height, dgraph.width, dgraph.height);
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
