#include <cuda.h>
#include "defines.h"
#include "coverkernel.h"

__global__ void kernCountCoverage(int *results, int *history, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, nfi, goffset,val, p_history, p_current;
	int *h, *r;
	int *rowResults, *row, *rowHistory;
	int *historyResults;
	if (tid < height) {
		rowResults = malloc(sizeof(int)*width);
		historyResults = malloc(sizeof(int)*width);
		r = malloc(sizeof(int)*2);
		h = malloc(sizeof(int)*2);
		for (int i = 0; i < width; i++) {
			rowResults[i] = 0;
			historyResults[i] = 0;
		}
		row = (int*)((char*)results + tid*(width)*sizeof(int));
		rowHistory = (int*)((char*)history + (tid+1)*(width)*sizeof(int));
		// need to calculate Np for both history and current
		for (int i = ncount; i >= 0; i--) {
			goffset = node[i].offset;
			nfi = node[i].nfi;
			if (node[i].po == 1) {
				val = fans[goffset];
				// if a transition is through this state and it's a PO, set = 1;
				rowResults[val] = row[val];
				historyResults[val] = results[val];
				h[1] = 0;
				h[0] = historyResults[fans[goffset]];
				v[0] = (rowResults[val] + historyResults[val]);
				v[1] = (rowResults[val]);
				rowResults[val] = r[(row[val] > rowHistory[val])];
				historyResults[val] = h[(row[val] > rowHistory[val])];
				continue;
			}
			// cover all logic gates
			for (int j = node[i].nfi - 1; j >= 0; j--) {
				val = fans[goffset+nfi-j];
				rowResults[val] = (row[fans[goffset+nfi]] >= 1)*rowResults[fans[goffset+nfi]];
				historyResults[val] = (rowHistory[fans[goffset+nfi]] >= 1)*rowHistory[fans[goffset+nfi]];
				h[1] = 0;
				h[0] = historyResults[val];
				v[0] = (rowResults[val] + historyResults[val]);
				v[1] = (rowResults[val]);
				rowResults[val] = r[(row[val] > rowHistory[val])];
				historyResults[val] = h[(row[val] > rowHistory[val])];
			}
		}
	}
}
float gpuCountPaths(ARRAY2D<int> results, ARRAY2D<int> history, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan) {
#ifndef NTIMING
	float elapsed = 0.0;
#endif

#ifndef NTIMING
	return elapsed;
#else
	return 0.0;
#endif
}
