#include "mergekernel.h"
#include <cuda.h>
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/segment.cuh"
#include "util/utility.cuh"
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
 inline uint32_t pred_gate(uint32_t a, uint32_t b) { return (((a) << 31) >> 31)&a; }
// Read the segment from the entry, determine the earliest pattern that marks it, and then write that (atomically).
// (likely) RESTRICTION: blockDim.x MUST be a power of 2
template <int N, int blockSize>
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
		if (threadIdx.x < 32) warpReduceMin(midWarp,threadIdx.x);
		if (threadIdx.x == 0) {
			// brief loop until we know that the item we wrote with AtomicMin is the correct positive minimum
			int2 candidate = midWarp[0];
			int2 evict = make_int2(-1,-1);

			evict.x = atomicMin(&(seglist[sid].pattern.x), (unsigned)(candidate.x));
			evict.y = atomicMin(&(seglist[sid].pattern.y), (unsigned)(candidate.y));

			printf("Lowest for this block: (%d, %d)", midWarp[0].x, midWarp[0].y);
		}
	}
	
}
/* Reduction strategy - X/1024 pattern blocks, Y blocks of lines/gates. Each
 * block gets the minimum ID within the block and places it into a temporary
 * location [BLOCK_X,BLOCK_Y] 
*/
__global__ void kernReduce(uint8_t* input, uint8_t* sim_input, size_t sim_pitch, size_t height, size_t pitch, int2* meta, uint32_t mpitch, uint32_t startGate, uint32_t startPattern) {
	uint32_t tid = threadIdx.x;
	startPattern += blockIdx.x*blockDim.x;
	uint32_t pid = tid + blockIdx.x*blockDim.x;
	uint32_t gid = blockIdx.y+startGate;
	int2 sdata;
	__shared__ int mx[32];
	__shared__ int my[32];
	uint8_t* row = input + pitch*gid;
	uint8_t* sim = sim_input + sim_pitch*gid;
	sdata = make_int2(-1,-1);
	if (tid < 32) {
		mx[tid] = -1;
		my[tid] = -1;
	}
	// Put the lower of pid and pid+MERGE_SIZE for which row[i] == 1
	// Minimum ID given by this is 1.
	if (pid < height) {

		// TODO: Run traverse_segments here for this segment, check to see if a segment has been marked for this block of threads. If not, abort. 

		int low_x = -1, low_y = -1;
		const uint8_t warp_id = threadIdx.x / (blockDim.x / 32); 
		unsigned int pred_x = (sim[pid] == T0)&&row[pid], pred_y = (sim[pid] == T1)&&row[pid];
		low_x = __ffs(__ballot(pred_x));
		low_y = __ffs(__ballot(pred_y));
		mx[warp_id] = (low_x > 0 ? low_x + (int32_t)startPattern + (warp_id * 32) : -1);
		my[warp_id] = (low_y > 0 ? low_y + (int32_t)startPattern + (warp_id * 32) : -1);
		// at this point, we have the position of the lowest.

#ifdef GPU_DEBUG
		printf("%s, %d: low[%d]: (%d,%d)\n", __FILE__, __LINE__, gid, low_x, low_y);
#endif

		__syncthreads();
		if (tid < 32) {
			pred_x = mx[tid] >= 0;
			pred_y = my[tid] >= 0;
			low_x = __ffs(__ballot(pred_x)) - 1;
			low_y = __ffs(__ballot(pred_y)) - 1;

			if (threadIdx.x == 0) {
				sdata.x = (low_x >= 0 ? mx[low_x] : -1)-1;
				sdata.y = (low_y >= 0 ? my[low_y] : -1)-1;
#ifdef GPU_DEBUG
				printf("%s, %d: low[%d]: (%d,%d)\n", __FILE__, __LINE__, gid, sdata.x, sdata.y);
#endif
				REF2D(meta,mpitch,blockIdx.x,blockIdx.y) = sdata; 
			}
		}

	}
}

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
		kernSegmentReduce<2,int2><<<blocks, MERGE_SIZE>>>(segs, toPod<coalesce_t>(sim), toPod<coalesce_t>(mark), segcount, startPattern);
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
float gpuMergeHistory(GPU_Data& input, GPU_Data& sim, void** mergeid, size_t chunk, uint32_t ext_startPattern) {
	if (*mergeid == NULL) { cudaMalloc(mergeid, sizeof(int2)*input.height()); } // only allocate a merge table on the first pass
	int2 *mergeids = (int2*)*mergeid;
	size_t count = 0;
	int2* temparray;
	size_t pitch;
	uint32_t startPattern = ext_startPattern;

//	uint32_t* debug = (uint32_t*)malloc(sizeof(uint32_t)*input.height());
//	uint32_t* debugt = (uint32_t*)malloc(sizeof(uint32_t)*input.height()*maxblock);
//	memset(debugt, 0, input.height()*maxblock);
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	// assume that the 0th entry is the widest, which is true given the chunking method.
#endif // NTIMING
	count = 0;
	size_t remaining_blocks = input.height();
	const size_t block_x = (input.gpu(chunk).width / MERGE_SIZE) + ((input.gpu(chunk).width % MERGE_SIZE) > 0);
	size_t block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
	cudaMallocPitch(&temparray, &pitch, sizeof(int2)*block_x, 65535);
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting on memory allocation
	do {
		dim3 blocks(block_x, block_y);
		kernReduce<<<blocks, MERGE_SIZE>>>(input.gpu(chunk).data, sim.gpu(chunk).data, sim.gpu(chunk).pitch, input.gpu(chunk).width, input.gpu(chunk).pitch, temparray, pitch, count, startPattern);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting inside the kernels
		dim3 blocksmin(1, block_y);
		kernSetMin<<<blocksmin, 1>>>(mergeids, temparray,  pitch, block_x, count, startPattern, chunk);
		count+=65535;
		if (remaining_blocks < 65535) { remaining_blocks = 0;}
		if (remaining_blocks > 65535) { remaining_blocks -= 65535; }
		block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
	} while (remaining_blocks > 0);
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting inside the kernels
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
