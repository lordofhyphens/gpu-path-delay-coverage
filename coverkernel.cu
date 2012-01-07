#include "defines.h"
#include "coverkernel.h"

void HandleCoverError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleCoverError( err, __FILE__, __LINE__ ))
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
#define REF2D(TYPE,ARRAY,PITCH,X,Y) ( (((TYPE*)((char*)ARRAY + Y*PITCH))[X] ))
#define ADDR2D(TYPE,ARRAY,PITCH,X,Y) ( (((TYPE*)((char*)ARRAY + Y*PITCH))+X ))
//Usage: REF2D(int,o_count,p_count,pid,gid)
#define GREF(GRAPH,SUB,OFFSET, X) ( GRAPH[SUB[OFFSET+X]] )
#define FIN(AR, OFFSET, ID) ( AR[OFFSET+ID] ) 
// reference: design book 1, page 38.
// lines -> 2d array of marked lines from marking kernel.
// p_lines -> pitch of lines array.
// history -> 1D array of identifiers stating which vector is the first with a '1'.
// o_count -> 2D array of path counts for each line, the one we're interested in
// o_hcount -> 2D array of path counts for the history data.
// start_gate -> gate ID from which to offset the block ID to get which gate these threads are working on
// max_pattern -> the highest pattern ID to work on. This will be either the total pattern count OR the current slice
__global__ void kernCountCoverage(GPUNODE* graph, char* lines,  size_t p_lines, int* fanins,int* history, int* o_count, size_t p_count, int* o_hcount, size_t p_hcount, int start_gate, int start_pattern, int max_pattern) {
	int tid = (blockIdx.y * blockDim.y) + threadIdx.x; // which pattern this thread is working on
	int pid = tid + start_pattern; 
	int gid = start_gate + blockIdx.x;
	int nfi, po, type, goffset;
	int val = 0; 
	if (tid < max_pattern) {
		int marked = REF2D(char,lines,p_lines,pid,gid);
		type = graph[gid].type;
		nfi = graph[gid].nfi;
		po = graph[gid].po;
		goffset = graph[gid].offset;
		if (po) {
			// Set the current paths to 1 if this line is marked and the history is not.
			REF2D(int,o_count,p_count,tid,gid) = (pid < history[gid])*(marked > 0);
			// Set the history count to 1 if this line is marked.
			REF2D(int,o_hcount,p_hcount,tid,gid) = (pid >= history[gid]);
		}
		if (type == FROM) {
			// Add the current paths to the fanin's count if this line is marked and the history is not.
			val = REF2D(int,o_count,p_hcount,pid,tid)*(REF2D(char,lines,p_lines,tid,gid) > 0)*(pid < history[gid]);
			atomicAdd(ADDR2D(int,o_count,p_count,pid,FIN(fanins,goffset,0)),val);
			// Add the current paths to the fanin's hcount if this line is marked.
			atomicAdd(ADDR2D(int,o_hcount,p_hcount,tid,FIN(fanins,goffset,0)), REF2D(int,o_hcount,p_hcount,tid,gid)*(pid >= history[gid]));
		} else { 
			// Assign current count to fan-ins if the fanin is marked and the history is.
			// Assign current history count to fan-ins if the fanin is marked.
			for (int i = 0; i < nfi; i++) {
				REF2D(int,o_count,p_count,tid,FIN(fanins,goffset,i)) = (marked > 0)*(pid < history[gid])*(REF2D(int,o_count,p_count,tid,gid));
				REF2D(int,o_hcount,p_hcount,tid,FIN(fanins,goffset,i)) = (pid >= history[gid])*(REF2D(int,o_hcount,p_hcount,tid,gid));
			}
		}
	}
}

float gpuCountPaths(ARRAY2D<char> input, ARRAY2D<int> count, ARRAY2D<int> hcount, ARRAY2D<int> history, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan, int maxlevels) {
	int *gatesinLevel, startGate=0;
	int *gpu_count, *gpu_hcount, *cpu_count, *cpu_hcount;
	size_t gpu_count_pitch, gpu_hcount_pitch;
	size_t free_memory = 0, used_memory, total_memory = 0;
	gatesinLevel = new int[maxlevels];
	size_t start_pattern = 0;
	for (int i = 0; i < maxlevels; i++) {
		gatesinLevel[i] = 0;
		for (unsigned int j = 0; j < input.width; j++) {
			if (graph[j].level == i) {
				gatesinLevel[i]++;
			}
		}
		startGate += gatesinLevel[i];
	}
	used_memory = dgraph.size() + input.size() + history.size();
	cudaMemGetInfo(&free_memory, &total_memory);
	DPRINT("Used memory: %lu, Free Memory: %lu, Total Memory: %lu\n",used_memory, free_memory, total_memory);
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	// figure out how much GPU memory we have left that is available, 
	int batch_row = (free_memory / sizeof(int)) / input.width; // remaining GPU memory in terms of # of patterns;
	if ((unsigned)batch_row > input.height) {
		batch_row = input.height*2;
	}
	batch_row /= 2;
	DPRINT("Allocating 2*%d columns of %lu rows. \n", batch_row, input.width);
	HANDLE_ERROR(cudaMallocPitch(&gpu_count, &gpu_count_pitch, sizeof(int)*batch_row, input.width));
	HANDLE_ERROR(cudaMallocPitch(&gpu_hcount, &gpu_hcount_pitch, sizeof(int)*batch_row, input.width));
	DPRINT("Actually allocated %lu bytes of memory, including pitch of %lu, width of %lu\n",input.width*gpu_count_pitch*2, gpu_count_pitch, sizeof(int)*batch_row);
	cpu_count = (int*)malloc(gpu_count_pitch*input.width);
	cpu_hcount = (int*)malloc(gpu_hcount_pitch*input.width);
	DPRINT("Starting pattern = %lu\n", start_pattern);
	// allocate two blocks of memory for that many rows*lines, one for history, one for actual results.
	DPRINT("History array width %lu, height %lu\n", history.width, history.height);
	do {
		int blockcount_y = (int)(batch_row/COVER_BLOCK) + (batch_row%COVER_BLOCK > 0);
		printf("yblocks: %d\n", blockcount_y);
		for (int i = maxlevels-1; i >= 0; i--) {
			dim3 numBlocks(gatesinLevel[i],blockcount_y);
			startGate -= gatesinLevel[i];
			kernCountCoverage<<<numBlocks,COVER_BLOCK>>>(dgraph.data, input.data, input.pitch, fan, history.data,gpu_count, gpu_count_pitch, gpu_hcount, gpu_hcount_pitch, startGate, start_pattern, batch_row);
			cudaDeviceSynchronize();
			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
		}
		// copy the patterns from 
		cudaMemcpy2D(cpu_count, gpu_count_pitch, gpu_count, gpu_count_pitch, sizeof(int)*batch_row, input.width, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(cpu_hcount, gpu_hcount_pitch, gpu_hcount, gpu_hcount_pitch, sizeof(int)*batch_row, input.width, cudaMemcpyDeviceToHost);
		start_pattern += batch_row;
		printf("Input.height %lu, start pattern: %lu\n", input.height, start_pattern);
	} while (start_pattern < input.height); 
	free(cpu_count);
	free(cpu_hcount);
	delete gatesinLevel;
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start,stop));
	return elapsed;
#else
	return 0.0;
#endif
}
