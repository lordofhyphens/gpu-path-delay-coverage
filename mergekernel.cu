#include "mergekernel.h"

void HandleMergeError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleMergeError( err, __FILE__, __LINE__ ))
#define LESS(A, B) ( sdata[A]*(sdata[A]<sdata[B])*(sdata[A] > 0) + sdata[A]*(sdata[B] == 0) + sdata[B]*(sdata[B]<sdata[A])*(sdata[B] > 0) + (sdata[B])*(sdata[A] == 0) )

__global__ void kernReduce(char* input, size_t height, size_t pitch, int goffset,int* meta, int mpitch) {
	int tid = threadIdx.x;
	__shared__ int sdata[MERGE_SIZE];
	char* row = input + pitch*(blockIdx.y+goffset);
	int* m = (int*)((char*)meta + mpitch*(blockIdx.y+goffset));
	unsigned int i = blockIdx.x*(MERGE_SIZE*2) + threadIdx.x;
	sdata[tid] = 0;
	// need to put the lower of i and i+MERGE_SIZE for which g_idata[i] == 1
	// Minimum ID given by this is 1.
	if (i < height) {
		if (i+MERGE_SIZE > height) { // correcting for blocks smaller than MERGE_SIZE
			sdata[tid] = (row[i] == 1)*(i+1);
		} else {
			sdata[tid] = (row[i] == 1)*(i+1) + (row[i+MERGE_SIZE] == 1)*(row[i] == 0)*(i+MERGE_SIZE+1);
		}
	}
	__syncthreads();

	// this is loop unrolling
    // do reduction in shared mem, comparisons against MERGE_SIZE are done at compile time.
    if (MERGE_SIZE >= 1024) { if (tid < 512) { sdata[tid] = LESS(tid, tid+512); } __syncthreads(); }
    if (MERGE_SIZE >= 512) { if (tid < 256) { sdata[tid] = LESS(tid, tid+256); } __syncthreads(); }
    if (MERGE_SIZE >= 256) { if (tid < 128) { sdata[tid] = LESS(tid, tid+128); } __syncthreads(); }
    if (MERGE_SIZE >= 128) { if (tid <  64) { sdata[tid] = LESS(tid, tid+64); } __syncthreads(); }
	if (tid < 32) {
		// Within a warp,  don't need __syncthreads();
		if (MERGE_SIZE >=  64) { sdata[tid] = LESS(tid, tid + 32); }
		if (MERGE_SIZE >=  32) { sdata[tid] = LESS(tid, tid + 16); }
		if (MERGE_SIZE >=  16) { sdata[tid] = LESS(tid, tid +  8); }
		if (MERGE_SIZE >=   8) { sdata[tid] = LESS(tid, tid +  4); }
		if (MERGE_SIZE >=   4) { sdata[tid] = LESS(tid, tid +  2); }
		if (MERGE_SIZE >=   2) { sdata[tid] = LESS(tid, tid +  1); }
	}
	// at this point, we have the position of the lowest. Correct by 1 to compensate for above.

//	if (tid ==0 ) { printf("Final Tid: %d, line %d, data+1 %d \n", tid, blockIdx.y, sdata[tid]); }
	if (threadIdx.x == 0) { m[blockIdx.x] = sdata[0]-1; }
	sdata[tid] = (sdata[0]-1)*(sdata[0]>0) + (sdata[0] == 0)*MERGE_SIZE*2;
	__syncthreads();
}

__global__ void kernSetMin(int* g_odata, size_t pitch,int* intermediate, int length, int i_pitch, int goffset) {
	unsigned int tid = (blockIdx.x*blockDim.x)+threadIdx.x;
	unsigned int gid = blockIdx.y;
	int* blockset = (int*)((char*)intermediate + (i_pitch*gid));
	if (tid == 0) { //first thread 
		// scan sequentially until a thread ID is discovered;
		int i = 0;
		while (blockset[i] < 0 && i < length) {
			i++;
		}
		if (i == length) {
			g_odata[gid] = -1;
		} else {
			g_odata[gid] = blockset[i];
		}
	}
	__syncthreads();
}
// scan through input until the first 1 is found, save the identifier and memset all indicies above that.
float gpuMergeHistory(ARRAY2D<char> input, ARRAY2D<int> mergeids) {
	size_t block_x = (input.height / MERGE_SIZE) + (input.height % MERGE_SIZE) > 1;
	size_t block_y = input.width;
	int* temparray;
	size_t pitch;
	cudaMallocPitch(&temparray, &pitch, sizeof(int)*block_x, block_y);
	dim3 blocks(block_x, block_y);
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	DPRINT("Blocks: (%lu, %lu), %d\n", block_x, block_y, MERGE_SIZE);
	kernReduce<<<blocks, MERGE_SIZE>>>(input.data, input.height, input.pitch, 0, temparray, pitch);
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
	dim3 blocksmin(1, block_y);
	kernSetMin<<<blocksmin, MERGE_SIZE>>>(mergeids.data, mergeids.pitch, temparray, block_x, pitch, 0);
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif
#ifndef NTIMING
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}