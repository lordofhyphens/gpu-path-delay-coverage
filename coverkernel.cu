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

// reference: design book 1, page 38.  Modified such that input_array is
// expected to be an array of integers containing, for each gate the first
// vector ID that is marked in the history.

__global__ void kernCountCoverage(GPUNODE* graph, char* input_array, size_t input_pitch, int* h_line, size_t history_pitch, size_t result_width, size_t pattern_count, int* count_array, size_t count_pitch, int hcount_array, size_t hcount_pitch, int* fanout_index, size_t width, size_t start_gate, size_t poffset) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x + poffset;
	__shared__ int nfi, goffset, po, type, fanouts[50];
	int gid = blockIdx.x + start_gate;
	char *line;
	int *count, *h_count;
	if (tid < pattern_count) {
		if (threadIdx.x == 0) {
			goffset = node[gid].offset;
			nfi = node[gid].nfi;
			po = node[gid].po;
			type = node[i].type;
			for (int i = 0; i < nfi; i++) {
				fanouts[i] = fanout_index[i+goffset];
			}
		}
		__syncthreads();

		line = ((char*)input_array + gid*input_pitch);
		count = (int*)((char*)count_array + gid*count_pitch);
		if (tid > 0) {
			h_count = ((char*)hcount_array + gid*hcount_pitch);
		} else {
			h_line = (char*)malloc(sizeof(char)*result_width);
			h_count = (int*)malloc(sizeof(int)*result_width)
		}
		if (po) {
			count = line[tid] && (tid < h_line[gid]); // only set = 1 if there's a new line here
			h_count = tid >= h_line[gid];
		}
		if (type == FROM) {
			atomicAdd(count[fanouts[gid]], count[gid]*(line[fanouts[0]])*(tid < h_line[fanouts[0]]));
			atomicAdd(h_count[fanouts[0]], h_count[gid]*(tid >= h_line[fanouts[0]]));
		} else if (type != INPT) {
			for (int fin = 0; fin < node[i].nfi; fin++) {
				// if the fanout total is 0 but this line is marked (and the history isn't), add a path to the count.
				// If the fanout total is > 1 and this line is marked (and the history isn't), assign the fanout total to the fanins.
				h_count[fanouts[fin]] = ((tid >= h_line[fanouts[fin]]) || h_count[tid] > 1) * h_count[tid];
				count[fanouts[fin]] = ((line[fanouts[fin]] && (tid < h_line[fanouts[fin]])) || count[tid] > 1) * count[tid] + h_count[tid]*(count[tid] == 0 && line[fanouts[fin]] > 1 && (tid < h_line[fanouts[fin]]);
			}
		}
		if (tid == 0) {
			free(h_line);
			free(h_count);
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
	int *gatesinLevel, startGate=0;
	gatesinLevel = new int[maxlevels];
	for (int i = 0; i < maxlevels; i++) {
		gatesinLevel[i] = 0;
		for (unsigned int j = 0; j < results.width; j++) {
			if (graph[j].level == i) {
				gatesinLevel[i]++;
			}
		}
		startGate += gatesinLevel[i];
	}
	int blockcount_y = (int)(input.height/THREAD_PER_BLOCK) + (input.height%THREAD_PER_BLOCK > 0);

#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	// figure out how much GPU memory we have
	for (int i = maxlevels-1; i >= 0; i--) {
		dim3 numBlocks(gatesinLevel[i],blockcount_y);
		startGate -= gatesinLevel[i];
		kernCountCoverage<<<numBlocks,THREAD_PER_BLOCK>>>(input.data, results.data, dgraph.data, fan, results.width, results.height, startGate, results.pitch);
		cudaDeviceSynchronize();
	}
	delete gatesinLevel;
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
