#include "mergekernel.h"
#include <cuda.h>

void HandleMergeError( cudaError_t err, const char *file, uint32_t line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleMergeError( err, __FILE__, __LINE__ ))

#define MIN(A,B,AR) ( \
		(AR[A] > 0)*(AR[A] < AR[B])*AR[A] + \
		(AR[B] > 0)*(AR[B] < AR[A])*AR[B] + \
		AR[A]*(AR[B]==0) + \
		AR[B]*(AR[A]==0) )

__global__ void kernReduce(uint8_t* input, size_t height, size_t pitch, int32_t* meta, uint32_t mpitch, uint32_t startGate) {
	uint32_t tid = threadIdx.x;
	uint32_t gid = blockIdx.y+startGate;
	__shared__ uint32_t sdata[MERGE_SIZE];
	uint8_t* row = input + pitch*gid;
	 uint32_t i = blockIdx.x*(MERGE_SIZE*2) + threadIdx.x;
	sdata[tid] = 0;
	// need to put the lower of i and i+MERGE_SIZE for which g_idata[i] == 1
	// Minimum ID given by this is 1.
	if (i < height) {
		if (i+MERGE_SIZE > height) { // correcting for blocks smaller than MERGE_SIZE
			sdata[tid] = (row[i] == 1)*(i+1);
			//printf("%s:%d - input[%d][%d] = %d\n", __FILE__,__LINE__,i, gid, row[i]);
		} else {
			sdata[tid] = (row[i] == 1)*(i+1) + (row[i+MERGE_SIZE] == 1)*(row[i] == 0)*(i+MERGE_SIZE+1);
		}
		__syncthreads();

		// this is loop unrolling
		// do reduction in shared mem, comparisons against MERGE_SIZE are done at compile time.
		if (MERGE_SIZE >= 1024) { if (tid < 512 && tid+512 < height) { sdata[tid] = MIN(tid, tid+512,sdata); } __syncthreads(); }
		if (MERGE_SIZE >= 512) { if (tid < 256 && tid+256 < height) { sdata[tid] = MIN(tid, tid+256,sdata); } __syncthreads(); }
		if (MERGE_SIZE >= 256) { if (tid < 128 && tid+128 < height) { sdata[tid] = MIN(tid, tid+128,sdata); } __syncthreads(); }
		if (MERGE_SIZE >= 128) { if (tid <  64 && tid+64 < height) { sdata[tid] = MIN(tid, tid+64,sdata); } __syncthreads(); }
		if (tid < 32) {
			// Within a warp,  don't need __syncthreads();
			if (MERGE_SIZE >=  64) { if (tid+32 < height) { sdata[tid] = MIN(tid, tid + 32,sdata); } }
			if (MERGE_SIZE >=  32) { if (tid+16 < height) { sdata[tid] = MIN(tid, tid + 16,sdata); } }
			if (MERGE_SIZE >=  16) { if (tid+8 < height) { sdata[tid] = MIN(tid, tid +  8,sdata); } }
			if (MERGE_SIZE >=   8) { if (tid+4 < height) { sdata[tid] = MIN(tid, tid +  4,sdata); } }
			if (MERGE_SIZE >=   4) { if (tid+2 < height) { sdata[tid] = MIN(tid, tid +  2,sdata); } }
			if (MERGE_SIZE >=   2) { if (tid+1 < height) { sdata[tid] = MIN(tid, tid +  1,sdata); } }
		}

		// at this point, we have the position of the lowest. Correct by 1 to compensate for above.

//		if (tid ==0 ) { printf("Final Tid: %d, line %d, data+1 %d \n", tid, blockIdx.y, sdata[tid] - 1); }
//		sdata[tid] = (sdata[0]-1)*(sdata[0]>0) + (sdata[0] == 0)*MERGE_SIZE*2;
		if (threadIdx.x == 0) { REF2D(uint32_t,meta,mpitch,blockIdx.x,gid) = sdata[0]-1; }
		__syncthreads();

	}
}

__global__ void kernSetMin(int32_t* g_odata, size_t pitch,int32_t* intermediate, uint32_t i_pitch,uint32_t length, uint32_t startGate) {
	 uint32_t gid = blockIdx.y + startGate;
	// scan sequentially until a thread ID is discovered;
	uint32_t i = 0;
	while (REF2D(int32_t, intermediate, i_pitch, i, gid) < 0 && i < length) {
		i++;
	}
//	printf("%s:%d - gid[%d] i = %d\n", __FILE__, __LINE__, gid, i);
	if (i < length) {
		g_odata[gid] = REF2D(int32_t, intermediate, i_pitch, i, gid);
	} else {
		g_odata[gid] = -1;
	}
//	printf("%s:%d - g_odata[%d] = %d\n", __FILE__, __LINE__, gid, g_odata[gid]);

}
// scan through input until the first 1 is found, save the identifier and memset all indicies above that.
float gpuMergeHistory(GPU_Data& input, ARRAY2D<int32_t>& mergeids) {
	size_t remaining_blocks = input.height();
	size_t maxblock = (input.width() / MERGE_SIZE) + ((input.width() % MERGE_SIZE) > 0);
	uint32_t count = 0;
	int32_t* temparray;
	size_t pitch;
	cudaMallocPitch(&temparray, &pitch, sizeof(int32_t)*maxblock, remaining_blocks);
	cudaMalloc(&mergeids.data, sizeof(int32_t)*input.height());
//	uint32_t* debug = (uint32_t*)malloc(sizeof(uint32_t)*input.height());
//	uint32_t* debugt = (uint32_t*)malloc(sizeof(uint32_t)*input.height()*maxblock);
//	memset(debugt, 0, input.height()*maxblock);
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	for ( uint32_t chunk = 0; chunk < input.size(); chunk++) {
		count = 0;
		size_t block_x = (input.gpu(chunk).width / MERGE_SIZE) + ((input.gpu(chunk).width % MERGE_SIZE) > 0);
		size_t block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
		do {
			//DPRINT("%s:%d - Blocks: %lu/%lu (%lu, %lu), %d\n", __FILE__, __LINE__, input.gpu(chunk).width, input.width(), block_x, block_y, MERGE_SIZE);
			dim3 blocks(block_x, block_y);
			kernReduce<<<blocks, MERGE_SIZE>>>(input.gpu(chunk).data, input.gpu(chunk).width, input.gpu(chunk).pitch, temparray, pitch, count);
			cudaDeviceSynchronize();
/*			cudaMemcpy2D(debugt, sizeof(uint32_t)*block_x, temparray, pitch, sizeof(uint32_t)*block_x, input.height(), cudaMemcpyDeviceToHost);
			for ( uint32_t j = 0; j < block_x/2; j++) {
				for ( uint32_t i = 0; i < input.height(); i++) {
					//DPRINT("%4d ", REF2D(uint32_t, debugt, sizeof(uint32_t)*block_x,j,i));
				}
				//DPRINT("\n");
			}
 */
			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
			dim3 blocksmin(1, block_y);
			DPRINT("Merging for %lu blocks of patterns", block_x);
			kernSetMin<<<blocksmin, 1>>>(mergeids.data, mergeids.pitch, temparray,  pitch, block_x/2, count);
			cudaDeviceSynchronize();
			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
			count+=65535;
			if (remaining_blocks > 65535) { remaining_blocks -= 65535; }
			block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
		} while (remaining_blocks > 65535);
/*		cudaMemcpy(debug, mergeids.data, sizeof(uint32_t)*input.height(),  cudaMemcpyDeviceToHost);
		for ( uint32_t i = 0; i < input.height(); i++) {
			//DPRINT("%2d ", debug[i]);
		}
		//DPRINT("\n");
 */
	}
	cudaFree(temparray);
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
void debugMergeOutput(ARRAY2D<int32_t>& results, std::string outfile) {
#ifndef NDEBUG
	int32_t *lvalues;
	std::ofstream ofile(outfile.c_str());
	ofile << "Size: " << results.width << "x" << results.height << " WxH "  << results.pitch << " Pitch" << std::endl;
	lvalues = (int32_t*)malloc(results.height*results.pitch);
	cudaMemcpy(lvalues,results.data,results.width*sizeof(int32_t),cudaMemcpyDeviceToHost);
	for (uint32_t r = 0;r < results.width; r++) {
		ofile << "Gate " << r << ":\t";
		for (uint32_t i = 0; i < results.height; i++) {
			int32_t z = lvalues[r];//REF2D(uint8_t, lvalues, results.pitch, r, i);
			switch(z) {
				default:
					ofile << std::setw(OUTJUST) << (int32_t)z << " "; break;
			}
		}
		ofile << std::endl;
	}
	free(lvalues);
	ofile.close();
#endif
}
