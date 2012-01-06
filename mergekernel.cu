#include "mergekernel.h"

void HandleMergeError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleMergeError( err, __FILE__, __LINE__ ))
#define BLOCK_SIZE 512
#define LESS(A, B) ( sdata[A]*(sdata[A]<sdata[B])*(sdata[A] > 0) + sdata[A]*(sdata[B] == 0) + sdata[B]*(sdata[B]<sdata[A])*(sdata[B] > 0) + (sdata[B])*(sdata[A] == 0) )

__global__ void kernReduce(char* input, char* results,size_t height, size_t pitch, int goffset,int* meta, int mpitch) {
	int tid = threadIdx.x;
	__shared__ int sdata[BLOCK_SIZE];
	char* row = input + pitch*(blockIdx.y+goffset);
	char* res = results + pitch*(blockIdx.y+goffset);
	int* m = (int*)((char*)meta + mpitch*(blockIdx.y+goffset));
	unsigned int i = blockIdx.x*(BLOCK_SIZE*2) + threadIdx.x;
	unsigned int mid = (blockIdx.x*blockDim.x) + threadIdx.x;
	sdata[tid] = 0;
	// need to put the lower of i and i+BLOCK_SIZE for which g_idata[i] == 1
	// Minimum ID given by this is 1.
	if (i < height) {
		if (i+BLOCK_SIZE > height) { // correcting for blocks smaller than BLOCK_SIZE
			sdata[tid] = (row[i] == 1)*(i+1);
		} else {
			sdata[tid] = (row[i] == 1)*(i+1) + (row[i+BLOCK_SIZE] == 1)*(row[i] == 0)*(i+BLOCK_SIZE+1);
		}
	}
	__syncthreads();

	// this is loop unrolling
    // do reduction in shared mem, comparisons against BLOCK_SIZE are done at compile time.
    if (BLOCK_SIZE >= 1024) { if (tid < 512) { sdata[tid] = LESS(tid, tid+512); } __syncthreads(); }
    if (BLOCK_SIZE >= 512) { if (tid < 256) { sdata[tid] = LESS(tid, tid+256); } __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) { sdata[tid] = LESS(tid, tid+128); } __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) { sdata[tid] = LESS(tid, tid+64); } __syncthreads(); }
	if (tid < 32) {
		// Within a warp,  don't need __syncthreads();
		if (BLOCK_SIZE >=  64) { sdata[tid] = LESS(tid, tid + 32); }
		if (BLOCK_SIZE >=  32) { sdata[tid] = LESS(tid, tid + 16); }
		if (BLOCK_SIZE >=  16) { sdata[tid] = LESS(tid, tid +  8); }
		if (BLOCK_SIZE >=   8) { sdata[tid] = LESS(tid, tid +  4); }
		if (BLOCK_SIZE >=   4) { sdata[tid] = LESS(tid, tid +  2); }
		if (BLOCK_SIZE >=   2) { sdata[tid] = LESS(tid, tid +  1); }
	}
	// at this point, we have the position of the lowest. Correct by 1 to compensate for above.

//	if (tid ==0 ) { printf("Final Tid: %d, line %d, data+1 %d \n", tid, blockIdx.y, sdata[tid]); }
	if (threadIdx.x == 0) { m[blockIdx.x] = sdata[0]-1; }
	sdata[tid] = (sdata[0]-1)*(sdata[0]>0) + (sdata[0] == 0)*BLOCK_SIZE*2;
	__syncthreads();
	if (mid >= sdata[tid]) {
		res[mid] = 1;
	}
}

__global__ void kernSetMin(char* results, size_t pitch,size_t pattern_count, int* intermediate, int i_pitch, int goffset) {
	unsigned int tid = (blockIdx.x*blockDim.x)+threadIdx.x;
	unsigned int gid = blockIdx.y;
	char* res = results + pitch*(blockIdx.y+goffset) + tid;

	__shared__ int index;
	int* blockset = (int*)((char*)intermediate + (i_pitch*gid));
	if (threadIdx.x == 0) { //first thread 
		// scan sequentially until a thread ID is discovered;
		int i = 0;
		while (blockset[i] < 0 && i < gridDim.x) {
			i++;
		}
		blockset[0] = blockset[i];
		index = i;
	}
	__syncthreads();
	if (tid >= index) {
		*res = 1;
	}
}
// scan through input until the first 1 is found, save the identifier and memset all indicies above that.
float gpuMergeHistory(ARRAY2D<char> input, ARRAY2D<char> mergeresult) {
	size_t block_x = (input.height / BLOCK_SIZE) + (input.height % BLOCK_SIZE) > 1;
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
	kernReduce<<<blocks, BLOCK_SIZE>>>(input.data, mergeresult.data, input.height, input.pitch, 0, temparray, pitch);
	cudaDeviceSynchronize();
	kernSetMin<<<blocks, BLOCK_SIZE>>>(mergeresult.data, mergeresult.pitch, mergeresult.height, temparray, pitch, 0);
	cudaDeviceSynchronize();
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
