#ifndef MERGEKERNEL_H
#define MERGEKERNEL_H
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/utility.h"

#include <cuda.h>
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/segment.cuh"
#include "util/utility.cuh"

#include <mgpuhost.cuh>
#include <mgpudevice.cuh>
#include <device/ctascan.cuh>
#include <device/intrinsics.cuh>
void HandleMergeError( cudaError_t err, const char *file, uint32_t line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
 }
 #undef HANDLE_ERROR
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
		for (int j = 0; j < N; j++) {
			ofile << z.key.num[j];
			if (j != N-1) 
				ofile << ",";
		}
		ofile << "): ";
		ofile << z.pattern.x << "," << z.pattern.y << " ";
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
const unsigned int PARALLEL_SEGS = 20;
const unsigned int WARP_SIZE = 32;
const unsigned int MERGE_SIZE = 1024;

__host__ __device__ __forceinline__ bool operator==(const int2 a, const int2 b) {
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


template <unsigned int blockSize>
__device__ __forceinline__ void warpReduceMin(volatile int2 sdata[PARALLEL_SEGS][WARP_SIZE], const unsigned int x, const unsigned int y) {
	if (blockSize >= 64) sdata[y][x].x = min((unsigned)(sdata[y][x].x), (unsigned)(sdata[y][x + 32].x));
	if (blockSize >= 32) sdata[y][x].x = min((unsigned)(sdata[y][x].x), (unsigned)(sdata[y][x + 16].x));
	if (blockSize >= 16) sdata[y][x].x = min((unsigned)(sdata[y][x].x), (unsigned)(sdata[y][x + 8].x));
	if (blockSize >= 8)  sdata[y][x].x = min((unsigned)(sdata[y][x].x), (unsigned)(sdata[y][x + 4].x));
	if (blockSize >= 4)  sdata[y][x].x = min((unsigned)(sdata[y][x].x), (unsigned)(sdata[y][x + 2].x));
	if (blockSize >= 2)  sdata[y][x].x = min((unsigned)(sdata[y][x].x), (unsigned)(sdata[y][x + 1].x));
	
	if (blockSize >= 64) sdata[y][x].y = min((unsigned)(sdata[y][x].y), (unsigned)(sdata[y][x + 32].y));
	if (blockSize >= 32) sdata[y][x].y = min((unsigned)(sdata[y][x].y), (unsigned)(sdata[y][x + 16].y));
	if (blockSize >= 16) sdata[y][x].y = min((unsigned)(sdata[y][x].y), (unsigned)(sdata[y][x + 8].y));
	if (blockSize >= 8)  sdata[y][x].y = min((unsigned)(sdata[y][x].y), (unsigned)(sdata[y][x + 4].y));
	if (blockSize >= 4)  sdata[y][x].y = min((unsigned)(sdata[y][x].y), (unsigned)(sdata[y][x + 2].y));
	if (blockSize >= 2)  sdata[y][x].y = min((unsigned)(sdata[y][x].y), (unsigned)(sdata[y][x + 1].y));
}
__device__ __forceinline__ unsigned int has_marked_segment(const GPU_DATA_type<coalesce_t> mark, const segment<2,int2> seg, const uint32_t pid) {
	unsigned int mark_set = 0xffffffff;
	if (seg.key.num[0] >= 0) mark_set &= REF2D(mark, pid, seg.key.num[0]);
	if (seg.key.num[1] >= 0) mark_set &= REF2D(mark, pid, seg.key.num[1]);
	return mark_set;
}
__device__ __forceinline__ unsigned int has_marked_segment(const GPU_DATA_type<coalesce_t> mark, const segment<1,int2> seg, const uint32_t pid) {
	return  0xffffffff & REF2D(mark, pid, seg.key.num[0]);
}

template <int N> 
__device__ __forceinline__ unsigned int has_marked_segment(const GPU_DATA_type<coalesce_t> mark, const segment<N,int2> seg, const uint32_t pid) {
	unsigned int mark_set = 0xffffffff;
	#pragma unroll
	for (uint8_t i = 0; i < N; i++) {
		// AND each mark result together.
		if (seg.key.num[i] >= 0) mark_set &= REF2D(mark, pid, seg.key.num[i]);
	}
	return mark_set;
}

__device__ void printState() {
}
// Read the segment from the entry, determine the earliest pattern that marks it, and then write that (atomically).
// (likely) RESTRICTION: blockDim.x MUST be a power of 2
template <int N, unsigned int blockSize>
__global__ void kernSegmentReduce2(segment<N, int2>* seglist, int maxSegment, const GPU_DATA_type<coalesce_t> mark, const GPU_DATA_type<coalesce_t> sim, uint32_t startSegment, const uint32_t startPattern) {
	__shared__ int2 midWarp[blockSize][WARP_SIZE];
	int2 final_value = make_int2(-1,-1);
	// Current segment for this thread group.
	midWarp[threadIdx.y][threadIdx.x] = make_int2(-1,-1);
	__syncthreads();
	unsigned int sid = startSegment + threadIdx.y + blockIdx.y*blockDim.y;
#pragma unroll 2
	for (int ref_pid = 0; ref_pid <= MERGE_SIZE; ref_pid += WARP_SIZE) { // progress in batches of WARP_SIZE. 
		// reduce to minimum for this warp, and store in final_value 
		uint32_t pid = threadIdx.x + ref_pid;
		uint32_t real_pid = pid * 4 + startPattern; // unroll constant for coalesce_t
		
		if (pid * 4 < mark.width && threadIdx.y + blockIdx.y*blockDim.y < maxSegment) {
			unsigned int mark_set = has_marked_segment(mark, seglist[sid], pid);

			// check to see which position got marked. This will be one of 4 possible positions:
			// 1, 9, 17, 25 (as returned by ffs).
			// Post brev:
			// 32 = offset 0
			// 24 = offset 1
			// 16 = offset 2 // 8 = offset 3
			// We OR the original set with itself, shifted and then mask off the garbage.

			unsigned int sim_type = REF2D(sim, pid, seglist[sid].key.num[0]) & 0x01010101;
			unsigned mark_set_x = mark_set & ~sim_type & 0x01010101;
			unsigned mark_set_y = mark_set & sim_type;
			
			// If the simulation results is T0 (or 2), then the real PID needs to be put into shared mem, x location
			// If the simulation results is T1 (or 3), then the real PID needs to be put into shared mem, y location
			////printf("real_pid %d sim_type %8x mark_set: %8x (%8x,%8x)\n",real_pid, sim_type, mark_set, mark_set_x,mark_set_y); 
			mark_set =  (mark_set | (mark_set >> 7) | (mark_set >> 14) | (mark_set >> 21)) & 0x0000000F;
			mark_set_x =  (mark_set_x | (mark_set_x >> 7) | (mark_set_x >> 14) | (mark_set_x >> 21)) & 0x0000000F;
			mark_set_y =  (mark_set_y | (mark_set_y >> 7) | (mark_set_y >> 14) | (mark_set_y >> 21)) & 0x0000000F;
			// reversing the bits puts the lowest bit first, 
			mark_set_x = __brev(mark_set_x);
			mark_set_y = __brev(mark_set_y);
			////printf("real_pid %d mark_set: %x (%x,%x)\n",real_pid, mark_set, mark_set_x,mark_set_y); 
			//which lets us get its position with ffs.
			unsigned int offset_x = (__clz(mark_set_x))+1;
			unsigned int offset_y = (__clz(mark_set_y))+1;
			
			midWarp[threadIdx.y][threadIdx.x].x = min((unsigned)midWarp[threadIdx.y][threadIdx.x].x, (unsigned)((offset_x+real_pid)*(offset_x<6)-1));
			
			midWarp[threadIdx.y][threadIdx.x].y = min((unsigned)midWarp[threadIdx.y][threadIdx.x].y, (unsigned)((offset_y+real_pid)*(offset_y < 6)-1));
			__syncthreads();
			if (sid == 70 && (pid == 27)) {
				//printf("hi there from pid thread %d in %d,%d !\n",pid, blockIdx.x,blockIdx.y);
				//printf("mark_set: %8X, %8X; offset %d, %d, representing %d, %d, %d, %d\n", mark_set_x, mark_set_y, offset_x ,offset_y, real_pid, real_pid+1,real_pid+2,real_pid+3);
				//printf("The minimum for this thread is located at offsets (%d,%d), real pids (%d,%d)\n", offset_x, offset_y, (offset_x+real_pid)*(offset_x<6)-1, (offset_y+real_pid)*(offset_y<6)-1);
				//printf("mark_set: midWarp[%d][%d] = (%d,%d)\n", threadIdx.y, threadIdx.x, midWarp[threadIdx.y][threadIdx.x].x, midWarp[threadIdx.y][threadIdx.x].y);
			}
			
			if (threadIdx.x < (WARP_SIZE / 2 )) 
				warpReduceMin<WARP_SIZE>(midWarp, threadIdx.x, threadIdx.y);

			__syncthreads();
			if (sid == 70 && (pid == 27)) {
				//printf("Minimum after reduction: (%d,%d)\n", midWarp[threadIdx.y][0].x, midWarp[threadIdx.y][0].y);
				//printf("MidWarp after: (%d,%d)\n", midWarp[threadIdx.y][threadIdx.x].x, midWarp[threadIdx.y][threadIdx.x].y);
			}
			if (threadIdx.x == 0) { 
				final_value.x = min((unsigned)midWarp[threadIdx.y][0].x,(unsigned)final_value.x); 
				final_value.y = min((unsigned)midWarp[threadIdx.y][0].y,(unsigned)final_value.y); 
			}
		}
	}
	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y + blockIdx.y*blockDim.y < maxSegment) {
		int2 candidate = make_int2(-1,-1);
		if (final_value.x >= 0) {
			while ((unsigned)final_value.x < (unsigned)candidate.x) {
				candidate.x = atomicMin((unsigned*)&(seglist[sid].pattern.x), (unsigned)(final_value.x));
			}
		}
		if (final_value.y >= 0) { 
			while ((unsigned)final_value.y < (unsigned)candidate.y) {
				candidate.y = atomicMin((unsigned*)&(seglist[sid].pattern.y), (unsigned)(final_value.y));
			}
		}
	}
}

template <int N>
float gpuMergeSegments(GPU_Data& mark, GPU_Data& sim, GPU_Circuit& ckt, size_t chunk, uint32_t ext_startPattern, void** seglist, int& numseg) {

	// if g_hashlist == NULL, copy the hash list to the GPU
//	ckt.print();
	segment<N, int2>* dc_seglist = (segment<N, int2>*)*seglist;
	uint32_t startPattern = ext_startPattern;
	uint32_t segcount = 0;
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	std::cerr << "Starting merge from " << ext_startPattern << "\n";
	if (dc_seglist == NULL) { 
		segment<N, int2>* h_seglist = NULL;
		generateSegmentList(&h_seglist,ckt);
		
		//displaySegmentList(h_seglist, ckt);
		while ((unsigned)h_seglist[segcount].key.num[0] < ckt.size()) { h_seglist[segcount].pattern.x = -1; h_seglist[segcount].pattern.y = -1; segcount++;}
		cudaMalloc(seglist, sizeof(segment<N,int2>)*segcount);
		dc_seglist = (segment<N, int2>*)*seglist;
		cudaMemcpy(dc_seglist, h_seglist, sizeof(segment<N,int2>)*segcount, cudaMemcpyHostToDevice);
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
	size_t remaining_blocks = (segcount / PARALLEL_SEGS) + ((segcount % PARALLEL_SEGS) > 0);
	const size_t block_x = (mark.gpu(chunk).width / MERGE_SIZE) + ((mark.gpu(chunk).width % MERGE_SIZE) > 0);
	size_t block_y = (remaining_blocks > BLOCK_STEP ? BLOCK_STEP : remaining_blocks);
	std::cerr << "Working with " << block_y * PARALLEL_SEGS<< " / " << remaining_blocks * PARALLEL_SEGS << " sids.\n"; 

	GPU_DATA_type<coalesce_t> marks = toPod<coalesce_t>(mark,chunk);
	GPU_DATA_type<coalesce_t> sims = toPod<coalesce_t>(sim,chunk);
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting on memory allocation
	do {
		dim3 threads(WARP_SIZE, PARALLEL_SEGS);
		dim3 blocks(block_x, block_y);
		kernSegmentReduce2<N,PARALLEL_SEGS><<<blocks, threads>>>(dc_seglist, segcount, marks, sims,count, startPattern);

		count+=BLOCK_STEP;
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting inside the kernels
		dim3 blocksmin(1, block_y);
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
if (verbose_flag && merge_flag) {
	std::cerr << "Printing merge log." << std::endl;
	debugSegmentList(dc_seglist, numseg, "gpumerge.log");
}
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
#endif
