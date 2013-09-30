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
#include <device/intrinsics.cuh>
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
void debugSegmentList(segment<N,int2>* seglist, const unsigned int& size, std::string outfile) {
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

const unsigned int BLOCK_STEP = 65535; // # of SIDs to process at once.

inline __host__ __device__ int2 min(int2 a, int2 b) { 
	return make_int2(min((unsigned)a.x, (unsigned)b.x), min((unsigned)a.y, (unsigned)b.y));
}
inline __host__ __device__ bool operator==(const int2& a, const int2&b) {
	return (a.x == b.x) && (a.y == b.y);
}

template <unsigned int blockSize>
__device__ void warpReduceMin(volatile int2 sdata[], unsigned int tid) {
	if (blockSize >= 64) sdata[tid].x = min((unsigned)sdata[tid].x, (unsigned)sdata[tid + 32].x);
	if (blockSize >= 64) sdata[tid].y = min((unsigned)sdata[tid].y, (unsigned)sdata[tid + 32].y); 
	if (blockSize >= 32) sdata[tid].x = min((unsigned)sdata[tid].x, (unsigned)sdata[tid + 16].x);
	if (blockSize >= 32) sdata[tid].y = min((unsigned)sdata[tid].y, (unsigned)sdata[tid + 16].y);
	if (blockSize >= 16) sdata[tid].x = min((unsigned)sdata[tid].x, (unsigned)sdata[tid + 8].x);
	if (blockSize >= 16) sdata[tid].y = min((unsigned)sdata[tid].y, (unsigned)sdata[tid + 8].y);
	if (blockSize >= 8)  sdata[tid].x = min((unsigned)sdata[tid].x, (unsigned)sdata[tid + 4].x);
	if (blockSize >= 8)  sdata[tid].y = min((unsigned)sdata[tid].y, (unsigned)sdata[tid + 4].y);
	if (blockSize >= 4)  sdata[tid].x = min((unsigned)sdata[tid].x, (unsigned)sdata[tid + 2].x);
	if (blockSize >= 4)  sdata[tid].y = min((unsigned)sdata[tid].y, (unsigned)sdata[tid + 2].y);
	if (blockSize >= 2)  sdata[tid].x = min((unsigned)sdata[tid].x, (unsigned)sdata[tid + 1].x);
	if (blockSize >= 2)  sdata[tid].y = min((unsigned)sdata[tid].y, (unsigned)sdata[tid + 1].y);
}
__host__ __device__ inline uint32_t pred_gate(uint32_t a, bool b) { return ((unsigned)(0-(!b))) &a; }
// Read the segment from the entry, determine the earliest pattern that marks it, and then write that (atomically).
// (likely) RESTRICTION: blockDim.x MUST be a power of 2
template <int N, unsigned int blockSize>
__global__ void kernSegmentReduce(segment<N, int2>* seglist, const GPU_DATA_type<coalesce_t> mark, const GPU_DATA_type<coalesce_t> sim, uint32_t startSegment, uint32_t startPattern) {
	__shared__ int2 midWarp[blockSize];
	midWarp[threadIdx.x] = make_int2(-1,-1);
	__syncthreads();

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
		#pragma unroll
		for (uint8_t i = 0; i < N; i++) {
			// AND each mark result together.
			mark_set &= REF2D(mark, pid, seglist[sid].key.num[i]);
		}

		// check to see which position got marked. This will be one of 4 possible positions:
		// 1, 9, 17, 25 (as returned by ffs).
		// Post brev:
		// 32 = offset 0
		// 24 = offset 1
		// 16 = offset 2 // 8 = offset 3
		// We OR the original set with itself, shifted and then mask off the garbage.
		mark_set =  (mark_set | (mark_set >> 7) | (mark_set >> 14) | (mark_set >> 20)) & 0x0000000F;
		// reversing the bits puts the lowest bit first, 
		mark_set = __brev(mark_set);
		//which lets us get its position with ffs.
		unsigned int offset = (32 - __ffs(mark_set));
		offset *= (offset < 5);

//		int sim_row = (uint32_t*)((char*)sim.data + sim.pitch*3)[2].r;

		// Figure out the relevant entry on the simualtion table.
		unsigned int sim_type = REF2D(sim, pid, seglist[sid].key.num[0]).rows[offset];
	//	printf("%d,%d: %8X\n",seglist[sid].key.num[0],  offset, sim_type);

		// If the simulation results is T0 (or 2), then the real PID needs to be put into shared mem, x location
		midWarp[threadIdx.x].x = (offset+real_pid+1)*((mark_set > 0)&&(sim_type == T0)) - 1;
		
		// If the simulation results is T1 (or 3), then the real PID needs to be put into shared mem, x location
		midWarp[threadIdx.x].y = (offset+real_pid+1)*((mark_set > 0)&&(sim_type == T1)) - 1;
		
		// actual PID + 1 we are comparing against or 0 if not found.
		// Place in shared memory, decrementing to correct real PID.


		__syncthreads();
		// Now do the reduction inside the same warp, looking for min-positive.
		if (blockSize >= 512) { if (threadIdx.x < 256) { midWarp[threadIdx.x] = min(midWarp[threadIdx.x], midWarp[threadIdx.x+256]); } __syncthreads();} 
		if (blockSize >= 256) { if (threadIdx.x < 128) { midWarp[threadIdx.x] =min(midWarp[threadIdx.x], midWarp[threadIdx.x+128]); } __syncthreads();} 
		if (blockSize >= 128) { if (threadIdx.x < 64) { midWarp[threadIdx.x] =min(midWarp[threadIdx.x], midWarp[threadIdx.x+64]); } __syncthreads();} 
		if (threadIdx.x < 32) warpReduceMin<blockSize>(midWarp,threadIdx.x);
			__syncthreads();
		if (threadIdx.x == 0) {
			// brief loop until we know that the item we wrote with AtomicMin is the correct positive minimum
			int2 candidate = midWarp[0];
			int2 evict = make_int2(-1,-1);

			do { evict.x = atomicMin(&(seglist[sid].pattern.x), (unsigned)(candidate.x));
			} while (min((unsigned)evict.x,(unsigned)candidate.x) == evict.x && evict.x != candidate.x);
			do { evict.y = atomicMin(&(seglist[sid].pattern.y), (unsigned)(candidate.y));
			} while (min((unsigned)evict.y,(unsigned)candidate.y) == evict.y && evict.y != candidate.y);

//			printf("Lowest for block %d: (%d, %d)\n", sid, midWarp[0].x, midWarp[0].y);
		}
	}
	
}
/* Reduction strategy - X/1024 pattern blocks, Y blocks of lines/gates. Each
 * block gets the minimum ID within the block and places it into a temporary
 * location [BLOCK_X,BLOCK_Y] 
*/

float gpuMergeSegments(GPU_Data& mark, GPU_Data& sim, GPU_Circuit& ckt, size_t chunk, uint32_t ext_startPattern, void** seglist, int& numseg) {

	// if g_hashlist == NULL, copy the hash list to the GPU
//	ckt.print();
	segment<2, int2>* dc_seglist = (segment<2, int2>*)*seglist;
	uint32_t startPattern = ext_startPattern;
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
	#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	// assume that the 0th entry is the widest, which is true given the chunking method.
#endif // NTIMING
	uint32_t count = 0;
	size_t remaining_blocks = segcount;
	const size_t block_x = (mark.gpu(chunk).width / MERGE_SIZE) + ((mark.gpu(chunk).width % MERGE_SIZE) > 0);
	size_t block_y = (remaining_blocks > BLOCK_STEP ? BLOCK_STEP : remaining_blocks);
	std::cerr << "Working with " << block_y << " / " << remaining_blocks << " sids.\n"; 

	GPU_DATA_type<coalesce_t> marks = toPod<coalesce_t>(mark,chunk);
	GPU_DATA_type<coalesce_t> sims = toPod<coalesce_t>(sim,chunk);
	debugMarkOutput(&marks, ckt, chunk, ext_startPattern, "gpumark-test.log");
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting on memory allocation
	do {
		dim3 blocks(block_x, block_y);
		kernSegmentReduce<2,MERGE_SIZE><<<blocks, MERGE_SIZE>>>(dc_seglist, marks, sims,count, startPattern);
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting inside the kernels
		dim3 blocksmin(1, block_y);
		count+=BLOCK_STEP;
		if (remaining_blocks < BLOCK_STEP) { remaining_blocks = 0;}
		if (remaining_blocks >= BLOCK_STEP) { remaining_blocks -= BLOCK_STEP; }
		block_y = (remaining_blocks > BLOCK_STEP ? BLOCK_STEP : remaining_blocks);
		cudaDeviceSynchronize();
	} while (remaining_blocks > 0);
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting inside the kernels
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif
#ifdef LOGEXEC
	debugSegmentList(dc_seglist, numseg, "gpumerge.log");
#endif //LOGEXEC
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
