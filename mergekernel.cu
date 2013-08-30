#include "mergekernel.h"
#include <cuda.h>
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/lookup.h"
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
__host__ __device__ inline uchar2 operator*=(const uchar2 &lhs, const bool &rhs) {
	return make_uchar2(lhs.x*rhs, lhs.y*rhs);
}
// "optimized" for single-threaded access,
template <int N>
__device__ void insertIntoHash(segment_t<N>* storage, const hashfuncs hashfunc,  hashkey<N> key, int2 data) {
	uint8_t hashes = 0;
	segment_t<N> evict; // hopefully we won't need this.
	evict.key.k[0] = 1;
	// repeat the insertion process while the thread still has an item.
	// Performance will be lower on non-Kepler architectures.
	// due to less efficient memory atomics.
	// If there is a collision, evict and try again with the next hash.
	int pred = 1;
	while (evict.key.k[0] != 0) {
		uint32_t hash = getHash<N>(key,hashfunc.g_hashlist[hashes],hashfunc.slots);
		//printf("Trying to set mutex for position %d, (%d, %d), in hashmap, which is currnetly %d\n",hash, key.k[0], key.k[1], storage[hash].mutex );
		while (pred && hashes < hashfunc.max) {
			pred = 1;
			while (pred != 0) { // spinlock, hate it but what can we do?
				pred = atomicCAS(&(storage[hash].mutex), 0, 1) != 0;
			}
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
template <int N> 
__device__ inline void insertIntoHash(segment_t<N>* storage, const hashfuncs hashfunc, segment_t<N> tgt) { insertIntoHash(storage,hashfunc,tgt.key, tgt.pid);}
// For a single block, determine the lowest tid for high/low, and return it.
__device__ int2 devSingleReduce(uchar2 pred, const uint32_t& startPattern, const uint32_t& pid, volatile int* mx, volatile int* my) {
	int2 sdata = make_int2(-1,-1);

	int32_t low_x = __ffs(__ballot(pred.x));
	int32_t low_y = __ffs(__ballot(pred.y));
	const uint8_t warp_id = threadIdx.x / (blockDim.x / 32); 

	// place lowest valid ID for x/y 
	mx[warp_id] = (low_x > 0 ? low_x + (int32_t)startPattern + (warp_id * 32) : -1);
	my[warp_id] = (low_y > 0 ? low_y + (int32_t)startPattern + (warp_id * 32) : -1);
	// at this point, we have the position of the lowest in the local warp.
	__syncthreads(); // should be safe.
	if (threadIdx.x < 32) {
		pred.x = mx[threadIdx.x] >= 0;
		pred.y = my[threadIdx.x] >= 0;
		low_x = __ffs(__ballot(pred.x)) - 1;
		low_y = __ffs(__ballot(pred.y)) - 1;

		if (threadIdx.x == 0) {
			sdata.x = (low_x >= 0 ? mx[low_x] : -1)-1;
			sdata.y = (low_y >= 0 ? my[low_y] : -1)-1;
		}
		mx[threadIdx.x] = -1;
		my[threadIdx.x] = -1;
	}
	if (threadIdx.x == 0) 
		printf("Reached end of reduction.\n");
	return sdata;

}

/* 
Stack storage for recursive function
Array of int2. 
	x=>the relative fanout for that level
	y=>ID of that fanout
	So stck[0].x points to the next level used of the 0th gate in the stack, and stck[0].y is the unique ID for thate gate.
	Tree descent:
		Move to next NFO immediately if available and not at end of segment.
		if at segment end, reduce, write segment, modify stack pointer
	
	Modify stack pointer:
		If pointer for current level is at end of available NFOs, reset current pointer to 0, decrement level
	Possible optimization: 
		skip tree if current gate is not marked. Requires a broadcast to all other nodes.
*/
template <int N>
__global__ void kernSegmentReduce(const g_GPU_DATA sim, const  g_GPU_DATA mark, uint32_t startGate, const GPUCKT ckt, uint32_t startPattern, segment_t<N>* hashmap, const hashfuncs hashfunc) {
	// Initial gate, used to mark off of others
	uint32_t gid = blockIdx.y+startGate;
	uint32_t pid = threadIdx.x + (blockIdx.x*blockDim.x);
	__shared__ int mx[32];
	__shared__ int my[32];
	__shared__ int2 stck[N];
	__shared__ int skip;
	int level = 0; 
	uchar2 pred = make_uchar2(0,0);

	if (pid < sim.width) { // TODO: Ensure that this is the correct dimension
		if (threadIdx.x == 0)
			printf("Starting on gid %d, level %d\n",gid, level);
		segment_t<N> scratch;
		// end once the pointer moves past the NFOs 
		stck[level].x = 0;
		__syncthreads();
		while (level >= 0) {
			// get current graph reference
			const GPUNODE& g = ckt.graph[gid];
			if (level == 0) {
				// figure out whether or not to start at the first or second
				pred.x = (REF2D(uint8_t, mark.data, mark.pitch, pid, gid) > 0 && REF2D(uint8_t, sim.data, sim.pitch, pid, gid) == T0);
				pred.y = (REF2D(uint8_t, mark.data, mark.pitch, pid, gid) > 0 && REF2D(uint8_t, sim.data, sim.pitch, pid, gid) == T1);
			}
			if (stck[level].x >= g.nfo) {
				if (threadIdx.x == 0)
					printf("Not a legal NFO. Aborting.\n");
				level--;
				continue;
			}	
			if (threadIdx.x == 0) {
				stck[level].y = FIN(ckt.fanout,g.offset,stck[level].x+g.nfi);
				printf("Working on gate %d, level %d. (%d,%d); NFO[%d]: %d\n",gid,level,stck[level].x,stck[level].y, gid,g.nfo);
			}
			__syncthreads();
			
			// place current gid in possible segment, current level.
			scratch.key.k[level] = gid;
			// end of segment
			// criteria: current g is a po, or the current level
			// is N-1 (counting from 0, so segment of length 2 would have
			// levels 0, 1).
			pred *= (REF2D(uint8_t, mark.data, mark.pitch, pid, gid) > 0);
			if (g.po == 1 || level >= N-1 || skip == 1) {
				// reduce and put into hash.
				// TODO: include reduction step
				if (!skip) {
					scratch.pid = devSingleReduce(pred,startPattern,pid,mx,my);				// we know to roll back a level if we are on the highest valid ID
					if (threadIdx.x == 0) { // only thread 0 has the lowest value here
						// place the scratch value into hash IFF it has a lower value than something else with the same key.
						printf("Inserting segment (%d,%d) into hashmap\n", scratch.key.k[0],scratch.key.k[1]);
						insertIntoHash<N>(hashmap,hashfunc,scratch);
					}

				}
				// for NFOs (counting from 0). 
				if (threadIdx.x == 0)
					stck[level].x += 1;
				__syncthreads();
				if (stck[level].x < g.nfo && skip == 0) {
					// everyone writes the same value
					stck[level].y = FIN(ckt.fanout,g.offset,g.nfi+stck[level].x);
				} else { 
					// rolling back a level
					// reset "stack pointer" to 0 for current level.
					stck[level].x = 0;
					stck[level].y = 0;
					// move back to previous level
					level--;
					// retrieve previous cached GID and increment NFO count.
					if (level >= 0)
						gid = stck[level].y; // retrieve the GID cached that refers to the parent.
					if (threadIdx.x == 0) {
						stck[level].x += 1; // increment the previous level's nfo pointer
					}
					__syncthreads();
					continue;
					// normal case, 
				}
				__syncthreads();
			}
			uint32_t tmp_gid = gid; // cache current GID.
			gid = stck[level].y; // set gid to the next 
			stck[level].y = tmp_gid;
			if (g.po != 1 && level < N-1) {
				level++;
			}
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

float gpuMergeSegments(GPU_Data& mark, GPU_Data& sim, GPU_Circuit& ckt, size_t chunk, uint32_t ext_startPattern, hashfuncs& dc_h, void** hash, uint32_t hashsize) {
#ifndef NTIMING
	float elapsed;
	void* tmp_hash = *hash;
	uint32_t startPattern = ext_startPattern;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	// assume that the 0th entry is the widest, which is true given the chunking method.
#endif // NTIMING
	// if g_hashlist == NULL, copy the hash list to the GPU
	if (dc_h.g_hashlist == NULL) {
		cudaMalloc(&(dc_h.g_hashlist),sizeof(uint32_t)*dc_h.max);
		cudaMemcpy(dc_h.g_hashlist, dc_h.h_hashlist, sizeof(uint32_t)*dc_h.max,cudaMemcpyHostToDevice);
	}
	if (*hash == NULL) { 
		cudaMalloc(&tmp_hash, sizeof(segment_t<2>)*hashsize(21));
		cudaMemset(tmp_hash, 0, sizeof(segment_t<2>)*hashsize(21));
		std::cerr << "Allocating hashmap space of " << hashsize(21) << ".\n";
	}
	segment_t<2>* hashmap = (segment_t<2>*)tmp_hash;
	uint32_t count = 0;
	size_t remaining_blocks = mark.height();
	const size_t block_x = (mark.gpu(chunk).width / MERGE_SIZE) + ((mark.gpu(chunk).width % MERGE_SIZE) > 0);
	size_t block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
//	block_y = 1; // only do one gid for testing
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting on memory allocation
	do {
		dim3 blocks(block_x, block_y);
		kernSegmentReduce<2><<<blocks, MERGE_SIZE>>>(toPod(sim), toPod(mark), count, toPod(ckt), startPattern, hashmap, dc_h);
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting inside the kernels
		dim3 blocksmin(1, block_y);
		count+=65535;
		if (remaining_blocks < 65535) { remaining_blocks = 0;}
		if (remaining_blocks > 65535) { remaining_blocks -= 65535; }
		block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
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
