#include "mergekernel.h"
#include <cuda.h>
#include "util/segment.h"
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/lookup.h"
#undef MERGE_SIZE
#undef LOGEXEC
#define MERGE_SIZE 1024
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
		((L) > 0)*((L) < (R))*(L) + \
		((R) > 0)*((R) < (L))*(R) )
#undef GPU_DEBUG
template <int N>
__device__ __host__ inline uint32_t getHash(const segment_t<N>& I, const uint32_t seed, uint8_t bits) {
	return hashlittle(I.key.h,sizeof(uint32_t)*N, seed) & hashmask(bits);
}
template <int N>
__device__ __host__ inline uint32_t getHash(const hashkey<N>& I, const uint32_t seed, uint8_t bits) {
	return hashlittle(I.h,sizeof(uint32_t)*N, seed) & hashmask(bits);
}
template <int N>
__device__ __host__ inline bool operator==(const hashkey<N>& lhs, const hashkey<N>&rhs) {
	bool tmp = true;
	#pragma unroll 2
	for (int i = 0; i < N; i++)
		tmp = tmp && (lhs.k[i] == rhs.k[i]);
	return tmp;
}
template <int N>
__device__ __host__ inline bool operator==(const hashkey<N>& lhs, const int &rhs) {
	bool tmp = true;
	#pragma unroll 2
	for (int i = 0; i < N; i++)
		tmp = tmp && (lhs.k[i] == rhs);
	return tmp;
}
template <int N>
__device__ __host__ inline bool operator!=(const hashkey<N>& lhs, const hashkey<N>&rhs) {
	return !(lhs == rhs);
}
typedef struct hashfuncs_t { uint32_t *hashlist, max, slots; } hashfuncs;
__device__ __host__ hashfuncs make_hashfuncs(uint32_t* hashlist, int max, int slots) { hashfuncs a; a.hashlist = hashlist, a.max = max, a.slots = slots; return a;}
template <int N>
__device__ bool isLegal(const segment_t<N>& segment) {
	return (segment.key == 0);
}
template <int N>
__device__ void insertIntoHash(segment_t<N>* storage, const hashfuncs hashfunc, const uint32_t slots, hashkey<N> key, int2 data) {
	uint8_t hashes = 0;
	segment_t<N> evict; // hopefully we won't need this.
	evict.key.k[0] = 1;
	// repeat the insertion process while the thread still has an item.
	// Performance will be lower on non-Kepler architectures.
	// due to less efficient memory atomics.
	// If there is a collision, evict and try again with the next hash.
	int pred = 1;
	while (evict.key.k[0] != 0) {
		uint32_t hash = getHash<N>(key,hashfunc.hashlist[hashes],hashfunc.slots);
		while (pred && hashes < hashfunc.max) {
			pred = 1;
			while (pred != 0) // spinlock, hate it but what can we do?
				pred = atomicCAS(&(storage[hash].mutex), 0, 1) != 0;
			if (key != storage[hash].key && isLegal<N>(storage[hash].key)) { // divergence possibility here.
				hashes++; // try another hash function.
			}
			// Loop until every thread gets a mutex on something,
			// or we run out of functions.
		}
		if (pred) { 
			// Something is seriously wrong here. It may be possible that the mutexing 
			// thread will give it up and we'll boot it out eventually.
			printf("Somehow, we managed to have %d hash collisions with concurrent executions and ran out of hash functions. Trying again.\n", N);
			hashes = 0;
			continue;
		}
		// Value of hash should be different in different threads. 
		// We're allowed to write to the structure now.
		evict = storage[hash]; evict.mutex = 0;
		if (evict.key == key) {
			for (int j = 0; j < N; j++) {
				storage[hash].pid.x = (evict.pid.x >= 0 && data.x >= 0 ? min(evict.pid.x, data.x) : max(evict.pid.x,data.x));
				storage[hash].pid.y = (evict.pid.y >= 0 && data.y >= 0 ? min(evict.pid.y, data.y) : max(evict.pid.y,data.y));
			}
		}
		else {
			storage[hash].key = key;
			storage[hash].pid = data;
		}
		atomicExch(&(storage[hash].mutex),0); // reset the mutex on this entry in memory.
		// check to see whether or not the eviction booted out a legal value that was not this key.
		// if it has, need to boot it out.
		if (evict.key != key && isLegal(evict.key)) {
			key = evict.key; data = evict.pid;
			hashes++;
		}
		if (hashes >= hashfunc.max) {
		 	// We ran out of hashes, reset? 
			// Probably a terrible idea.
			hashes = 0;
		}
	}
}
// For a single block, determine the lowest tid for high/low, and return it.
__device__ int2 devSingleReduce(uint32_t pred_x, uint32_t pred_y, const uint32_t& startPattern, const uint32_t& pid, volatile int* mx, volatile int* my) {
	int2 sdata = make_int2(-1,-1);

	int32_t low_x = __ffs(__ballot(pred_x));
	int32_t low_y = __ffs(__ballot(pred_y));
	const uint8_t warp_id = threadIdx.x / (blockDim.x / 32); 

	// place lowest valid ID for x/y 
	mx[warp_id] = (low_x > 0 ? low_x + (int32_t)startPattern + (warp_id * 32) : -1);
	my[warp_id] = (low_y > 0 ? low_y + (int32_t)startPattern + (warp_id * 32) : -1);
	// at this point, we have the position of the lowest in the local warp.
	__syncthreads(); // should be safe.
	if (threadIdx.x < 32) {
		pred_x = mx[threadIdx.x] >= 0;
		pred_y = my[threadIdx.x] >= 0;
		low_x = __ffs(__ballot(pred_x)) - 1;
		low_y = __ffs(__ballot(pred_y)) - 1;

		if (threadIdx.x == 0) {
			sdata.x = (low_x >= 0 ? mx[low_x] : -1)-1;
			sdata.y = (low_y >= 0 ? my[low_y] : -1)-1;
		}
		mx[threadIdx.x] = -1;
		my[threadIdx.x] = -1;
	}

	return sdata;

}

template <int N>  
__device__ void devSegmentRecurse(segment_t<N>& result, uint8_t level, uint32_t start_gid, const g_GPU_DATA& sim_info, const g_GPU_DATA& mark_info, const GPUCKT& ckt, const uint32_t& pid, bool cont, uint32_t x, uint32_t y, const uint32_t& startPattern, volatile int* mx, volatile int* my, segment_t<N>* hashmap, const hashfuncs& hashfunc, const uint32_t& slots) {
	if (level < N) {
		const GPUNODE& g = ckt.graph[start_gid];
		const uint8_t* row = mark_info.data + mark_info.pitch*start_gid;
		const uint8_t* sim = sim_info.data + sim_info.pitch*start_gid;
		uint32_t pred_x = (sim[pid] == T0)&&row[pid], pred_y = (sim[pid] == T1)&&row[pid];
		result.key.k[level] = start_gid;

		for (uint16_t i = 0; i < g.nfo; i++) { // traverse over the NFOs, recursing
			devSegmentRecurse(result, level+1, 
			FIN(ckt.fanout,g.offset,g.nfi+i), sim_info, mark_info, ckt, pid, cont && (__any(pred_x) || __any(pred_y)), pred_x, pred_y, startPattern, mx, my, hashmap, hashfunc,slots);
		}
		if (g.po == 1) { 
			level = N; 
			devSegmentRecurse(result, N, start_gid, sim_info, mark_info, ckt, pid, cont && (__any(pred_x) || __any(pred_y)), pred_x, pred_y, startPattern, mx, my, hashmap,hashfunc,slots);
		} // if this is a PO, not going any further
	} else {
		// reduce and place into hashmap if cont == true
		result.pid = devSingleReduce(x,y, startPattern, pid, mx, my);
		// only thread 0 of this block has the correct pid.
		if (threadIdx.x == 0) 
			insertIntoHash<N>(hashmap, hashfunc, slots, result.key, result.pid);
		// it's stored now, so reset local pid value before returning.
		result.pid.x = -1;
		result.pid.y = -1;
	}
	result.key.k[level] = 0; // unwinding stack, so need to reset.
}
template <int N>
__global__ void kernSegmentReduce(const g_GPU_DATA sim, const g_GPU_DATA mark, uint32_t startGate, const GPUCKT ckt, uint32_t startPattern, segment_t<N>* hashmap, const hashfuncs& hashfunc, const uint32_t slots) {
	// Initial gate, used to mark off of others
	uint32_t gid = blockIdx.y+startGate;
	uint32_t pid = threadIdx.x + (blockIdx.x*blockDim.x);
	__shared__ int mx[32];
	__shared__ int my[32];
	if (pid < sim.width) { // TODO: Ensure that this is the correct dimension
		segment_t<N> a;
		a.pid.x = -1;
		a.pid.y = -1;
		// recurse
		devSegmentRecurse(a, 0, gid, sim, mark, ckt, pid, true, 0, 0, startPattern, mx, my, hashmap, hashfunc, slots);
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
				REF2D(int2,meta,mpitch,blockIdx.x,blockIdx.y) = sdata; 
			}
		}

	}
}

__global__ void kernSetMin(int2* g_odata, int2* intermediate, uint32_t i_pitch, uint32_t length, uint32_t startGate, uint32_t startPattern, uint16_t chunk) {
	 uint32_t gid = blockIdx.y + startGate;
	// scan sequentially until a thread ID is discovered;
	int32_t i = 0;
	int32_t j = 0;
	while (REF2D(int2, intermediate, i_pitch, i, blockIdx.y).x < 0 && i < length) {
		i++;
	}
	while (REF2D(int2, intermediate, i_pitch, j, blockIdx.y).y < 0 && j < length) {
		j++;
	}
	if (i < length) {
		if (chunk > 0) {
			i = (g_odata[gid].x == -1 ? REF2D(int2, intermediate, i_pitch, i, blockIdx.y).x : g_odata[gid].x);
		} else {
			i = REF2D(int2, intermediate, i_pitch, i, blockIdx.y).x;
		}
	} else {
		if (chunk == 0) {
			i = -1;
		} else {
			i = g_odata[gid].x;
		}
	}
#ifdef GPU_DEBUG
	printf("%s, %d: g_odata[%d]: (%d,%d)\n", __FILE__, __LINE__, gid,i, j);
#endif
	if (j < length) {
		if (chunk > 0) {
			if (g_odata[gid].y >= 0) {
				j = g_odata[gid].y;
			} else {
				j = REF2D(int2, intermediate, i_pitch, j, blockIdx.y).y;
			}
		} else {
					j = REF2D(int2, intermediate, i_pitch, j, blockIdx.y).y;
		}
	} else {
			if (chunk == 0) {
			j = -1;
		} else {
			j = g_odata[gid].y;
		}
	}
#ifdef GPU_DEBUG
	printf("%s, %d: g_odata[%d]: (%d,%d)\n", __FILE__, __LINE__, gid,i, j);
#endif
	g_odata[gid].x = i;
	g_odata[gid].y = j;

}

float gpuMergeSegments(GPU_Data& mark, GPU_Data& sim, GPU_Circuit& ckt, size_t chunk, uint32_t ext_startPattern) {
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	// assume that the 0th entry is the widest, which is true given the chunking method.
#endif // NTIMING
	count = 0;
	size_t remaining_blocks = mark.height();
	const size_t block_x = (mark.gpu(chunk).width / MERGE_SIZE) + ((mark.gpu(chunk).width % MERGE_SIZE) > 0);
	size_t block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
	cudaMallocPitch(&temparray, &pitch, sizeof(int2)*block_x, 65535);
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting on memory allocation
	do {
		dim3 blocks(block_x, block_y);
		kernReduce<<<blocks, MERGE_SIZE>>>(mark.gpu(chunk).data, sim.gpu(chunk).data, sim.gpu(chunk).pitch, mark.gpu(chunk).width, mark.gpu(chunk).pitch, temparray, pitch, count, startPattern);
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
