#include "mergekernel.h"
#include <cuda.h>
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/segment.cuh"
#include "util/utility.cuh"
#include "markkernel.h"

#include <mgpuhost.cuh>
#include <mgpudevice.cuh>
#include <device/ctascan.cuh>
#undef N
#undef MERGE_SIZE
#define MERGE_SIZE 512
void HandleMergeError( cudaError_t err, const char *file, uint32_t line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
 }
#define HANDLE_ERROR( err ) (HandleMergeError( err, __FILE__, __LINE__ ))

template<int N>
void debugSegmentList(segment<N,int2>* seglist, const int& size, std::string outfile) {
	#ifndef NDEBUG
	segment<N,int2> *lvalues;
	std::ofstream ofile(outfile.c_str());
	lvalues = new segment<N,int2>[size];
	cudaMemcpy(lvalues,seglist,size*sizeof(segment<N,int2>),cudaMemcpyDeviceToHost);
	for (size_t r = 0;r < size; r++) {
		ofile << "Segment " << r << "(" ;
		segment<N,int2> z = lvalues[r];//REF2D(uint8_t, lvalues, results.pitch, r, i);
		#pragma unroll
		for (int j = 0; j < N; j++) {
			ofile << z.key.num[j];
			if (j != N-1) 
				ofile << ",";
		}
		ofile << "):\t";
		ofile << std::setw(OUTJUST) << z.pattern.x << "," << z.pattern.y << " ";
		ofile << std::endl;
		}
	delete lvalues;
	ofile.close();
#endif
}
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

const unsigned int BLOCK_STEP = 1; // # of SIDs to process at once.

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
__global__ void kernSegmentReduce(segment<N, int2>* seglist, const GPU_DATA_type<coalesce_t> mark, const GPU_DATA_type<coalesce_t> sim, uint32_t startSegment, uint32_t startPattern) {
	__shared__ int2 midWarp[blockSize];

	uint32_t pid = threadIdx.x + blockIdx.x*blockDim.x;
	uint32_t real_pid = pid * 4 + startPattern; // unroll constant for coalesce_t
	pid += startPattern;
	uint32_t sid = blockIdx.y+startSegment;
	int2 simple = make_int2(-1,-1);
	if (real_pid < mark.width) {
		// In each thread, get 4 results-worth. 
		// This should unroll, as the trip count is known at compile time.
		// Get a batch of mark and sim results
		unsigned int mark_set = 0xffffffff;
		for (uint8_t i = 0; i < N; i++) {
			// AND each mark result together.
			uint32_t z = 0;
			uint32_t gid = seglist[sid].key.num[i];
			mark_set &= ((uint32_t*)((unsigned char*)mark.data + (mark.pitch*gid)))[pid];
			printf("%8x, %d\n",((uint32_t*)((unsigned char*)mark.data + (mark.pitch*gid)))[pid],mark.pitch );
		}

		// check to see which position got marked. This will be one of 4 possible positions:
		// 1, 9, 17, 25 (as returned by ffs).
		// Post brev:
		// 32 = offset 0
		// 24 = offset 1
		// 16 = offset 2 // 8 = offset 3
		int offset = 5 - (__ffs(__brev(mark_set)) >> 3);
		int this_pid = pred_gate((offset < 5), real_pid + offset); // 
		printf("%d - %d,%d \n",pid, this_pid-1, offset);
		// actual PID + 1 we are comparing against or 0 if not found.
		// Place in shared memory, decrementing to correct real PID.
		midWarp[threadIdx.x].x = pred_gate((vectAND(REF2D(sim,threadIdx.x,seglist[sid].key.num[0]), T0) > 0), this_pid) - 1;
		midWarp[threadIdx.x].y = pred_gate((vectAND(REF2D(sim,threadIdx.x,seglist[sid].key.num[0]), T1) > 0), this_pid) - 1;

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

			printf("Lowest for block %d: (%d, %d)\n", sid, midWarp[0].x, midWarp[0].y);
		}
	}
	
}
/* Reduction strategy - X/1024 pattern blocks, Y blocks of lines/gates. Each
 * block gets the minimum ID within the block and places it into a temporary
 * location [BLOCK_X,BLOCK_Y] 
*/

float gpuMergeSegments(GPU_Data& mark, GPU_Data& sim, GPU_Circuit& ckt, size_t chunk, uint32_t ext_startPattern, void** seglist, int& numseg) {
#ifndef NTIMING
	float elapsed;
	segment<2, int2>* dc_seglist = (segment<2, int2>*)*seglist;
	uint32_t startPattern = ext_startPattern;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	// assume that the 0th entry is the widest, which is true given the chunking method.
#endif // NTIMING
	// if g_hashlist == NULL, copy the hash list to the GPU
//	ckt.print();
	uint32_t segcount = 0;
	if (dc_seglist == NULL) { 
		segment<2, int2>* h_seglist = NULL;
		generateSegmentList(&h_seglist,ckt);
		displaySegmentList(h_seglist, ckt);
		while (h_seglist[segcount].key.num[0] < ckt.size()) { segcount++;}
		cudaMalloc(&dc_seglist, sizeof(segment<2,int2>)*segcount);
		cudaMemcpy(dc_seglist, h_seglist, sizeof(segment<2,int2>)*segcount, cudaMemcpyHostToDevice);
		std::cerr << "Allocating hashmap space of " << segcount << ".\n";
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting inside the kernels
		numseg = segcount;
	} else { segcount = numseg; }
	uint32_t count = 0;
	size_t remaining_blocks = segcount;
	const size_t block_x = (mark.gpu(chunk).width / MERGE_SIZE) + ((mark.gpu(chunk).width % MERGE_SIZE) > 0);
	size_t block_y = (remaining_blocks > BLOCK_STEP ? BLOCK_STEP : remaining_blocks);
	std::cerr << "Working with " << block_y << " / " << remaining_blocks << " sids.\n"; 

	GPU_DATA_type<coalesce_t> marks = toPod<coalesce_t>(mark,chunk);
	std::cerr << "Podded pitch: " << marks.pitch << "\n";
	debugMarkOutput(&marks, ckt, chunk, ext_startPattern, "gpumark-test.log");
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting on memory allocation
	do {
		dim3 blocks(block_x, block_y);
		kernSegmentReduce<2,MERGE_SIZE><<<blocks, MERGE_SIZE>>>(dc_seglist, marks, toPod<coalesce_t>(sim,chunk),count, startPattern);
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting inside the kernels
		dim3 blocksmin(1, block_y);
		count+=BLOCK_STEP;
		if (remaining_blocks < BLOCK_STEP) { remaining_blocks = 0;}
		if (remaining_blocks >= BLOCK_STEP) { remaining_blocks -= BLOCK_STEP; }
		block_y = (remaining_blocks > BLOCK_STEP ? BLOCK_STEP : remaining_blocks);
		cudaDeviceSynchronize();
	} while (remaining_blocks > 0);
	cudaDeviceSynchronize();
#ifdef LOGEXEC
	debugSegmentList(dc_seglist, numseg, "gpumerge.log");
#endif //LOGEXEC
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
