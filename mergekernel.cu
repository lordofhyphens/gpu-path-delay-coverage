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
#undef MIN
#define MIN(A,B) ( \
		(A > 0)*(A < B)*A + \
		(B > 0)*(B < A)*B + \
		A*(B==0) + \
		B*(A==0) )
#define MIN_GEN(L,R) ( \
		((L) != 0)*((L) < (R))*(L) + \
		((R) != 0)*((R) < (L))*(R) )


__global__ void kernReduce(uint8_t* input, size_t height, size_t pitch, int2* meta, uint32_t mpitch, uint32_t startGate, uint32_t startPattern) {
	uint32_t tid = threadIdx.x;
	uint32_t gid = blockIdx.y+startGate;
	__shared__ int2 sdata[MERGE_SIZE];
	uint8_t* row = input + pitch*gid;
	 uint32_t i = blockIdx.x*(MERGE_SIZE*2)+tid;
	sdata[tid] = make_int2(0,0);
	// Put the lower of pid and pid+MERGE_SIZE for which row[i] == 1
	// Minimum ID given by this is 1.
	if (i < height) {
		if (i+MERGE_SIZE > height) { // correcting for blocks smaller than MERGE_SIZE
			sdata[tid] = make_int2((row[i] == T0)*(i+1),(row[i] == T1)*(i+1));
		} else {
			sdata[tid] = make_int2((row[i] == T0)*(i+1) + (row[i+MERGE_SIZE] == T0)*(row[i] == 0)*(i+MERGE_SIZE+1), (row[i] == T1)*(i+1) + (row[i+MERGE_SIZE] == T1)*(row[i] == 0)*(i+MERGE_SIZE+1));
		}
		__syncthreads();

		// this is loop unrolling
		// do reduction in shared mem, comparisons against MERGE_SIZE are done at compile time.
		if (MERGE_SIZE >= 1024) { if (tid < 512 && tid+512 < height) { sdata[tid] = make_int2(MIN(sdata[tid].x,sdata[tid + 512].x),MIN(sdata[tid].y,sdata[tid + 512].y)); } __syncthreads(); }
		if (MERGE_SIZE >= 512) { if (tid < 256 && tid+256 < height) { sdata[tid] = make_int2(MIN(sdata[tid].x,sdata[tid + 256].x),MIN(sdata[tid].y,sdata[tid + 256].y)); } __syncthreads(); }
		if (MERGE_SIZE >= 256) { if (tid < 128 && tid+128 < height) { sdata[tid] = make_int2(MIN(sdata[tid].x,sdata[tid + 128].x),MIN(sdata[tid].y,sdata[tid + 128].y)); } __syncthreads(); }
		if (MERGE_SIZE >= 128) { if (tid <  64 && tid+64 < height) { sdata[tid] = make_int2(MIN(sdata[tid].x,sdata[tid + 64].x),MIN(sdata[tid].y,sdata[tid + 64].y)); } __syncthreads(); }
		if (tid < 32) {
			// Within a warp,  don't need __syncthreads();
			if (MERGE_SIZE >=  64) { if (tid+32 < height) { sdata[tid] = make_int2(MIN(sdata[tid].x,sdata[tid + 32].x),MIN(sdata[tid].y,sdata[tid + 32].y)); } }
			if (MERGE_SIZE >=  32) { if (tid+16 < height) { sdata[tid] = make_int2(MIN(sdata[tid].x,sdata[tid + 16].x),MIN(sdata[tid].y,sdata[tid + 16].y)); } }
			if (MERGE_SIZE >=  16) { if (tid+8 < height) { sdata[tid] = make_int2(MIN(sdata[tid].x,sdata[tid + 8].x),MIN(sdata[tid].y,sdata[tid + 8].y));} }
			if (MERGE_SIZE >=   8) { if (tid+4 < height) { sdata[tid] = make_int2(MIN(sdata[tid].x,sdata[tid + 4].x),MIN(sdata[tid].y,sdata[tid + 4].y));} }
			if (MERGE_SIZE >=   4) { if (tid+2 < height) { sdata[tid] = make_int2(MIN(sdata[tid].x,sdata[tid + 2].x),MIN(sdata[tid].y,sdata[tid + 2].y));} }
			if (MERGE_SIZE >=   2) { if (tid+1 < height) { sdata[tid] = make_int2(MIN(sdata[tid].x,sdata[tid + 1].x),MIN(sdata[tid].y,sdata[tid + 1].y));} }
		}

		// at this point, we have the position of the lowest. Correct by 1 to compensate for above.

//		if (tid ==0 ) { printf("Final Tid: %d, line %d, data+1 %d \n", tid, blockIdx.y, sdata[tid] - 1); }
//		sdata[tid] = (sdata[0]-1)*(sdata[0]>0) + (sdata[0] == 0)*MERGE_SIZE*2;
		if (threadIdx.x == 0) { REF2D(int2,meta,mpitch,blockIdx.x,gid) = make_int2(sdata[0].x-1, sdata[0].y-1); }
		__syncthreads();

	}
}

__global__ void kernSetMin(int2* g_odata, size_t pitch, int2* intermediate, uint32_t i_pitch,uint32_t length, uint32_t startGate, uint32_t startPattern, uint16_t chunk) {
	 uint32_t gid = blockIdx.y + startGate;
	// scan sequentially until a thread ID is discovered;
	int32_t i = 0;
	int32_t j = 0;
	while (REF2D(int2, intermediate, i_pitch, i, gid).x < 0 && i < length) {
		i++;
	}
	while (REF2D(int2, intermediate, i_pitch, j, gid).y < 0 && j < length) {
		j++;
	}

	if (i < length) {
		if (chunk > 0) {
			i = MIN_GEN(REF2D(int2, intermediate, i_pitch, i, gid).x + startPattern, g_odata[gid].x);
		} else {
			i = REF2D(int2, intermediate, i_pitch, i, gid).x;
		}
	} else {
		if (chunk == 0) {
			i = -1;
		} else {
			i = g_odata[gid].x;
		}
	}
	if (j < length) {
		if (chunk > 0) {
			j = MIN_GEN(REF2D(int2, intermediate, i_pitch, j, gid).y + startPattern, g_odata[gid].y);
		} else {
			j = REF2D(int2, intermediate, i_pitch, j, gid).y;
		}
	} else {
		if (chunk == 0) {
			j = -1;
		} else {
			j = g_odata[gid].y;
		}
	}
	g_odata[gid] = make_int2(i,j);

}
// scan through input until the first 1 is found, save the identifier and memset all indicies above that.
float gpuMergeHistory(GPU_Data& input, void** mergeid) {

	cudaMalloc(mergeid, sizeof(int2)*input.height());
	int2 *mergeids = (int2*)*mergeid;
	size_t remaining_blocks = input.height();

	size_t maxblock = (input.width() / MERGE_SIZE) + ((input.width() % MERGE_SIZE) > 0);
	uint32_t count = 0;
	int2* temparray;
	size_t pitch;
	cudaMallocPitch(&temparray, &pitch, sizeof(int2)*maxblock, remaining_blocks);

//	uint32_t* debug = (uint32_t*)malloc(sizeof(uint32_t)*input.height());
//	uint32_t* debugt = (uint32_t*)malloc(sizeof(uint32_t)*input.height()*maxblock);
//	memset(debugt, 0, input.height()*maxblock);
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	uint32_t startPattern = 0;
#endif // NTIMING
	for ( uint32_t chunk = 0; chunk < input.size(); chunk++) {
		count = 0;
		size_t block_x = (input.gpu(chunk).width / MERGE_SIZE) + ((input.gpu(chunk).width % MERGE_SIZE) > 0);
		size_t block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
		do {
			dim3 blocks(block_x, block_y);
			kernReduce<<<blocks, MERGE_SIZE>>>(input.gpu(chunk).data, input.gpu(chunk).width, input.gpu(chunk).pitch, temparray, pitch, count, startPattern);
			cudaDeviceSynchronize();

			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
			dim3 blocksmin(1, block_y);
			kernSetMin<<<blocksmin, 1>>>(mergeids, input.height(), temparray,  pitch, (block_x/2) + (block_x/2 == 0), count, startPattern,chunk);
			cudaDeviceSynchronize();
			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
			count+=65535;
			if (remaining_blocks > 65535) { remaining_blocks -= 65535; }
			block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
		} while (remaining_blocks > 65535);
		startPattern += input.gpu(chunk).width;
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
void debugMergeOutput(size_t size, const void* res, std::string outfile) {
#ifndef NDEBUG
	int2 *lvalues, *results = (int2*)res;
	std::ofstream ofile(outfile.c_str());
	lvalues = new int2[size];
	cudaMemcpy(lvalues,results,size*sizeof(int2),cudaMemcpyDeviceToHost);
	for (size_t r = 0;r < size; r++) {
		ofile << "Gate " << r << ":\t";
		int2 z = lvalues[r];//REF2D(uint8_t, lvalues, results.pitch, r, i);
		ofile << std::setw(OUTJUST) << z.x << "," << z.y << " ";
		ofile << std::endl;
		}
	delete lvalues;
	ofile.close();
#endif
}
