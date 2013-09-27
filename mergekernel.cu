#include "mergekernel.h"
#include <cuda.h>
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/segment.cuh"
#include "util/utility.cuh"

#include <mgpuhost.cuh>
#include <mgpudevice.cuh>
#include <device/ctascan.cuh>
#undef N
#undef MERGE_SIZE
#undef LOGEXEC
#define MERGE_SIZE 512
void HandleMergeError( cudaError_t err, const char *file, uint32_t line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
 }
#define HANDLE_ERROR( err ) (HandleMergeError( err, __FILE__, __LINE__ ))

namespace mgpu {
	struct ScanOpMinPos {
		enum { Communative = true};
		typedef int2 input_type;
		typedef uint2 value_type;
		typedef int2 result_type;

		MGPU_HOST_DEVICE value_type Extract (value_type t, int index) { return t;}
		MGPU_HOST_DEVICE value_type Plus(value_type t1, value_type t2) { return make_uint2(min(t1.x,t2.x),min(t1.y,t2.x));};
		MGPU_HOST_DEVICE value_type Combine(value_type t1, value_type t2) { return t2; }
		MGPU_HOST_DEVICE value_type Identity() { return make_uint2((unsigned)_ident.x,(unsigned)_ident.y); }
		MGPU_HOST_DEVICE ScanOpMinPos(input_type ident) : _ident(ident) { }
		MGPU_HOST_DEVICE ScanOpMinPos() {
			_ident = make_int2(numeric_limits<int>::max(), numeric_limits<int>::max());
		}
		input_type _ident;
	};
}

__host__ __device__ inline int32_t min_pos(int32_t a, int32_t b) { return min((unsigned)a, (unsigned)b);}
__host__ __device__ inline int2 min_pos(int2 a, int2 b) { return make_int2(min((unsigned)a.x, (unsigned)b.x),min((unsigned)a.y,(unsigned)b.y));}

template <unsigned int blockSize>
__device__ void warpReduceMin(volatile int2 * sdata , unsigned int tid) {
	if (blockSize >= 64) min_pos(sdata[tid], sdata[tid + 32]) ; 
	if (blockSize >= 32) min_pos(sdata[tid], sdata[tid + 16]) ;
	if (blockSize >= 16) min_pos(sdata[tid],sdata[tid + 8]);
	if (blockSize >= 8) min_pos(sdata[tid], sdata[tid + 4]);
	if (blockSize >= 4) min_pos(sdata[tid], sdata[tid + 2]);
	if (blockSize >= 2) min_pos(sdata[tid], sdata[tid + 1]);
}
__host__ __device__ inline uint32_t pred_gate(uint32_t a, uint32_t b) { return (((a) << 31) >> 31)&a; }
// Read the segment from the entry, determine the earliest pattern that marks it, and then write that (atomically).
// (likely) RESTRICTION: blockDim.x MUST be a power of 2
template <int N, unsigned int blockSize>
__global__ void kernSegmentReduce(segment<N>* seglist, const GPU_DATA_type<coalesce_t> mark, const GPU_DATA_type<coalesce_t> sim, uint32_t startSegment, uint32_t startPattern) {
	__shared__ int2 midWarp[blockSize];

	uint32_t pid = threadIdx.x + blockIdx.x*blockDim.x;
	uint32_t real_pid = pid * 4 + startPattern; // unroll constant for coalesce_t
	uint32_t sid = blockIdx.y+startSegment;
	const uint32_t warp_id = (threadIdx.x >> 5);// threadId / 32 should be the warp #. 32 = 2^5. 
	int2 simple = make_int2(-1,-1);
	if (pid < mark.height) {
		startPattern += blockIdx.x*blockDim.x;
		// In each thread, get 4 results-worth. 
		// This should unroll, as the trip count is known at compile time.
		// Get a batch of mark and sim results
		coalesce_t mark_set = REF2D(mark, threadIdx.x, seglist[sid].key.num[0]);
		#pragma unroll
		for (uint8_t i = 1; i < N; i++) {
			// AND each mark result together.
			mark_set.packed &= REF2D(mark, threadIdx.x, seglist[sid].key.num[i]).packed;
		}
		// check to see which position got marked. This will be one of 4 possible positions:
		// 1, 9, 17, 25 (as returned by ffs).
		// Post brev:
		// 32 = offset 0
		// 24 = offset 1
		// 16 = offset 2 // 8 = offset 3
		int offset = 5 - (__ffs(__brev(mark_set.packed)) >> 3);
		int this_pid = pred_gate((offset < 5), real_pid + offset); // 
		// actual PID + 1 we are comparing against or 0 if not found.
		// Place in shared memory, decrementing to correct real PID.
		midWarp[threadIdx.x].x = pred_gate((REF2D(sim,threadIdx.x,seglist[sid].key.num[0]).packed & T0 > 0), this_pid) - 1;
		midWarp[threadIdx.x].y = pred_gate((REF2D(sim,threadIdx.x,seglist[sid].key.num[0]).packed & T1 > 0), this_pid) - 1;

		// Now do the reduction inside the same warp, looking for min-positive.
		if (blockSize >= 512) { if (threadIdx.x < 256) { min_pos(midWarp[threadIdx.x], midWarp[threadIdx.x+256]); } __syncthreads();} 
		if (blockSize >= 256) { if (threadIdx.x < 128) { min_pos(midWarp[threadIdx.x], midWarp[threadIdx.x+128]); } __syncthreads();} 
		if (blockSize >= 128) { if (threadIdx.x < 64) { min_pos(midWarp[threadIdx.x], midWarp[threadIdx.x+64]); } __syncthreads();} 
		if (threadIdx.x < 32) warpReduceMin<blockSize>(midWarp,threadIdx.x);
		if (threadIdx.x == 0) {
			// brief loop until we know that the item we wrote with AtomicMin is the correct positive minimum
			int2 candidate = midWarp[0];
			int2 evict = make_int2(-1,-1);

			evict.x = atomicMin(&(seglist[sid].pattern.x), (unsigned)(candidate.x));
			evict.y = atomicMin(&(seglist[sid].pattern.y), (unsigned)(candidate.y));

			printf("Lowest for this block: (%d, %d)\n", midWarp[0].x, midWarp[0].y);
		}
	}
	
}
/* Reduction strategy - X/1024 pattern blocks, Y blocks of lines/gates. Each
 * block gets the minimum ID within the block and places it into a temporary
 * location [BLOCK_X,BLOCK_Y] 
*/

#define BLOCK_STEP 1
float gpuMergeSegments(GPU_Data& mark, GPU_Data& sim, GPU_Circuit& ckt, size_t chunk, uint32_t ext_startPattern, void** seglist) {
#ifndef NTIMING
	float elapsed;
	segment<2>* tmp_seglist = (segment<2>*)*seglist;
	uint32_t startPattern = ext_startPattern;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	// assume that the 0th entry is the widest, which is true given the chunking method.
#endif // NTIMING
	// if g_hashlist == NULL, copy the hash list to the GPU
	ckt.print();
	if (*seglist == NULL) { 
		cudaMalloc(&tmp_seglist, sizeof(segment<2>)*300);
		cudaMemset(tmp_seglist, 0, sizeof(segment<2>)*300);
		std::cerr << "Allocating hashmap space of " << 300 << ".\n";
	}
	segment<2>* segs = (segment<2>*)tmp_seglist;
	uint32_t segcount = 0;
	uint32_t count = 0;
	size_t remaining_blocks = mark.height();
	const size_t block_x = (mark.gpu(chunk).width / MERGE_SIZE) + ((mark.gpu(chunk).width % MERGE_SIZE) > 0);
	size_t block_y = (remaining_blocks > BLOCK_STEP ? BLOCK_STEP : remaining_blocks);
//	block_y = 1; // only do one gid for testing
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting on memory allocation
	do {
		dim3 blocks(block_x, block_y);
		kernSegmentReduce<2,MERGE_SIZE><<<blocks, MERGE_SIZE>>>(segs, toPod<coalesce_t>(sim), toPod<coalesce_t>(mark), segcount, startPattern);
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting inside the kernels
		dim3 blocksmin(1, block_y);
		count+=BLOCK_STEP;
		if (remaining_blocks < BLOCK_STEP) { remaining_blocks = 0;}
		if (remaining_blocks > BLOCK_STEP) { remaining_blocks -= BLOCK_STEP; }
		block_y = (remaining_blocks > BLOCK_STEP ? BLOCK_STEP : remaining_blocks);
		cudaDeviceSynchronize();
	} while (remaining_blocks > 0);
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting inside the kernels
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
// scan through input until the first 1 is found, save the identifier and set all indicies above that.
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
