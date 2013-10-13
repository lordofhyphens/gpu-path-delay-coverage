
#ifndef COVERKENREL_H
#define COVERKERNEL_H
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/defines.h"
#include "util/utility.h"
#include "util/segment.cuh"
#include "util/g_utility.h"
#include <cuda.h>

#undef OUTJUST
#define OUTJUST 4

void HandleCoverError( cudaError_t err, const char *file, uint32_t line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#undef LOGEXEC
#ifdef HANDLE_ERROR
#undef HANDLE_ERROR
	#define HANDLE_ERROR( err ) (HandleCoverError( err, __FILE__, __LINE__ ))
#endif


const unsigned int COVER_BLOCK = 512;
const unsigned int REDUCE_THREADS = 512;


//template <typename T>
//__device__ __forceinline__ T clip(const T in, const T low, const  T high) { return in <= low ? high : in >= high ? high : in -low; }
__device__ __forceinline__ int clip(const int in, const int2 bound) {
	return in < bound.x ? bound.y+1 - bound.x : in > bound.y ? bound.y+1 - bound.x : in - bound.x;
}
__device__ __forceinline__ int clip(const int in, const int2 bound, bool test) {
	return in < bound.x && !test ? 0 : in > bound.y && !test ? 0 : in - bound.x;
}
// bit masks for packing, trying to reduce # of fetches.
#define GET_H(SRC) (((SRC & 0xFFFF0000) >> 16))
#define GET_C(SRC) ((SRC & 0x0000FFFF))
#define PACK(H, C) ((H << 16) | (C & 0x0000FFFF))
void debugCover(const Circuit& ckt, uint32_t *cover, size_t patterns, size_t lines, std::ofstream& ofile, size_t chunk = 0, size_t startPattern = 0) {
#ifndef NDEBUG
	if (chunk == 0) {
		std::cerr << "Patterns: " << patterns << "; Lines: " << lines << std::endl;
		ofile << "Gate:   \t";
		int i = 0;
		while (ckt.at(i).typ == INPT) {
			ofile << std::setw(OUTJUST) << i++ << " ";
		}
		ofile << "\n";
	}
	for (uint32_t r = 0; r < patterns; r++) {
		ofile << "Vector " << r+startPattern << ":\t";
		for (uint32_t i = 0; i < lines; i++) {
			if (ckt.at(i).typ == INPT) {
				const uint32_t z = GET_C(REF2D(cover, sizeof(uint32_t)*patterns, r, i));
				ofile << std::setw(OUTJUST) << z << " ";
			}
		}
		ofile << "\n";
	}
#endif
}
template <uint16_t blockSize>
__device__ void warpReduce(volatile uint32_t* sdata, uint16_t tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void kernSumSingle(GPUNODE* ckt, size_t size, uint32_t* input, size_t height, size_t pitch, uint64_t* meta) {
	__shared__ uint32_t sdata[blockSize]; // only add positive #s
	uint16_t tid = threadIdx.x;
	sdata[tid] = 0; // reset shared data segment to 0.
	uint32_t local_count = 0;
	for (size_t i = 0; i < size; i++) { // iterate over everything in the circuit
		if (ckt[i].type != INPT) 
			continue; // short-circuit for non-PIs
		#pragma unroll 8
		for (size_t j = 0; j < height; j+=blockSize) { // unrolled to handle reductions of up to BLOCK_SIZE in parallel
			if (tid+j < height) {
				local_count += GET_C(REF2D(input, pitch, tid+j, i));
			}
		}
	}
	sdata[tid] = local_count; __syncthreads();
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256];  } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128];  } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64];    } __syncthreads();}
	if (tid < 32) { warpReduce<blockSize>(sdata, tid); } 
	__syncthreads();
	if (threadIdx.x == 0) {
		*meta += sdata[0];
	}
	__syncthreads();

}
__device__ __forceinline__ bool in_range(const int test, const int2 bound) {
	return (test >= bound.x && test <= bound.y);
}
__device__ __forceinline__ bool in_sid(const segment<2, int2> item, const uint32_t g) {
	return (item.key.num[0] == g || item.key.num[1] == g);
}
__device__ __forceinline__ bool in_sid(const segment<3, int2> item, const uint32_t g) {
	return (item.key.num[0] == g || item.key.num[1] == g|| item.key.num[2] == g);
}
template <int N>
__device__ __forceinline__ bool in_sid(const segment<N, int2> item, const uint32_t g) {
	bool test = false;
#pragma unroll
	for (int i = 0; i < N; i++) {
		//printf("G %d is in SID %d: %d\n", g, sid,(haystack[sid].key.num[i] == g));
		test = test || (item.key.num[i] == g);
	}
	return test;
}


template <int N, int blockSize>
__device__ void findActiveThreads(volatile int2 sdata[blockSize], segment<N,int2> const * const haystack, size_t haystack_size, const uint32_t g, const int2 bounds) {
// Divide haystack into chunks, each thread will work on haystack/blockSize threads
// search each chunk for G.
// if found and max >= pattern.x >= min; sdata[pattern.x - min] = pattern.x;
// if found and max >= pattern.y >= min; sdata[pattern.y - min] = pattern.y;
/*	if (threadIdx.x == 0) {
		printf("Bounds: (%d,%d)\n", bounds.x, bounds.y);
 		printf("Shared Memory size: %d\n", blockSize);
	}
 */
	// march through all of the segments, looking for ones that match the current GID.
	for (int batch_needle = 0; batch_needle < haystack_size; batch_needle += blockSize) { 
		int sid = batch_needle + threadIdx.x;
		if (sid < haystack_size) {
			assert(sid < haystack_size);
			bool test = in_sid(haystack[sid], g);
			// while this loop-unrolls, it's not very efficient.
			//			printf("sdata: %p, indirect %p, garbage: %p\n", sdata, indirect, &garbage);
			// current problem is that this diverges like mad.
			// test will be true if our gate shows up in the segment!
			if (test) {
				sdata[clip(haystack[sid].pattern.x,bounds)].x = haystack[sid].pattern.x;
				sdata[clip(haystack[sid].pattern.y,bounds)].y = haystack[sid].pattern.y;
			}
		}
	}
}

template <>
__device__ void findActiveThreads<1,COVER_BLOCK>(volatile int2 sdata[COVER_BLOCK], segment<1,int2> const * const haystack, size_t haystack_size, const uint32_t g, const int2 bounds) {
// Divide haystack into chunks, each thread will work on haystack/blockSize threads
// search each chunk for G.
// if found and max >= pattern.x >= min; sdata[pattern.x - min] = pattern.x;
// if found and max >= pattern.y >= min; sdata[pattern.y - min] = pattern.y;
/*	if (threadIdx.x == 0) {
		printf("Bounds: (%d,%d)\n", bounds.x, bounds.y);
 		printf("Shared Memory size: %d\n", blockSize);
	}
*/
	// march through all of the segments, looking for ones that match the current GID.
	sdata[threadIdx.x].x = haystack[g].pattern.x;
	sdata[threadIdx.x].y = haystack[g].pattern.y;
}
template <int N, int blockSize>
__global__ void kernCover(GPUNODE const * const ckt, uint8_t const * const mark, const size_t mark_pitch, segment<N,int2> const * const history, size_t history_size,  uint32_t* cover, const size_t cover_pitch, const uint32_t start_offset,const uint32_t pattern_count,const uint32_t start_pattern, uint32_t const * const offsets) { //, uint32_t* subckt, size_t subckt_size) {
    // cover is the coverage ints we're working with for this pass.
	// mark is the fresh marks
	// hist is the history of the mark status of all lines.
	const uint32_t tid = (blockIdx.y * COVER_BLOCK) + threadIdx.x;
	__shared__ int2 activeTests[blockSize+2];
	int2 threadBound; // min/max thread

	const int32_t pid = tid + start_pattern; 
	activeTests[threadIdx.x].x = -1;
	activeTests[threadIdx.x].y = -1;
	const uint32_t g = blockIdx.x+start_offset;
	const GPUNODE& gate = ckt[g];

	threadBound.x = start_pattern + (blockIdx.y * blockSize); threadBound.y = start_pattern + ((blockIdx.y+1) * (blockSize));
	// all threads participate here
	findActiveThreads<N,blockSize>(activeTests, history, history_size, g, threadBound);
	__syncthreads();

	// activeTests[threadIdx.x] should contain a PID if and only if G is in the segment and that segment is marked 

	if (tid < pattern_count) {
		const uint8_t cache = REF2D(mark,mark_pitch,tid, g); // cache the current node's marked status.
		// shorthand references to current coverage and history count.
		uint32_t c = 0, h = 0;

		if (gate.po == 1) {
			c = 0;
			h = (cache > 0); // set history = 1 if this line is marked.
		}

		uint32_t resultCache = 0, histCache = 0;
		for (uint32_t i = 0; i < gate.nfo; i++) {
			const uint32_t fot = FIN(offsets,gate.offset,gate.nfi+i);
			const uint32_t tmp = REF2D(cover, cover_pitch, tid, fot);
			resultCache += (GET_C(tmp)); // add this fanout's path count to this node.
			histCache   += (GET_H(tmp)); // add this fanout's history path count to this node.
		}
		assert(histCache <= 0xFFFF);
		assert(resultCache <= 0xFFFF);
		c += resultCache;
		h += histCache;
		{
			// c equals c+h if either activeTests[threadIdx.x] >= pid and line is marked
			const uint32_t tmp = h*(cache > 0)*((activeTests[threadIdx.x].x == pid) || (activeTests[threadIdx.x].y == pid));
			c = c*(cache > 0) + tmp;
			// h equals 0 if neither activeTests[threadIdx.x] >= pid, else h if this line is marked;
			h = h*(cache > 0)*((activeTests[threadIdx.x].x != pid)*(activeTests[threadIdx.x].y != pid));
		}
		if (gate.type == INPT) {
			h = 0; // history results at this point on input gates are useless.
		}
		assert (c <= 0xFFFF); // make sure everything fits into 16 bits
		assert (h <= 0xFFFF); // make sure everything fits into 16 bits
		// Cycle through the fanins of this node and assign them the current value
		const uint32_t tmp = PACK(h,c);
		REF2D(cover     , cover_pitch , tid, g) = tmp;
	}
}


template <int N>
float gpuCountPaths(const GPU_Circuit& ckt, GPU_Data& mark, const void* merge, const int& merge_size,
		uint64_t* coverage, size_t chunk, size_t startPattern) {

	segment<N,int2>* merges = (segment<N,int2>*)merge;

	HANDLE_ERROR(cudaGetLastError()); // check to make sure there aren't any errors going into function.

	std::ofstream cfile;
	if (verbose_flag) {
		if (chunk == 0) {
			cfile.open("gpucover.log", std::ios::out);
		} else {
			cfile.open("gpucover.log", std::ios::out | std::ios::app);
		}
	}
	uint32_t *g_results;
	uint32_t *d_results = NULL; // debug results 
	uint64_t *finalcoverage;
	uint32_t startGate;
	size_t pitch;
	//	const size_t summedPatterns = (mark.width() / (MERGE_SIZE*2)) + ((mark.width() % (MERGE_SIZE*2)) > 0);

	cudaMalloc(&finalcoverage, sizeof(uint64_t));
	cudaMemset(finalcoverage, 0, sizeof(uint64_t)); // set value to 0 explicitly
	HANDLE_ERROR(cudaGetLastError()); // checking that last memory operation completed successfully.

#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif
	uint32_t pcount = 0;
	gpuCheckMemory();
	DPRINT("Allocating for chunk %lu \n", chunk);
	DPRINT("Allocating %lu bytes for results... ",sizeof(uint32_t)*mark.gpu(chunk).width*mark.height());
	cudaMallocPitch(&g_results,&pitch, sizeof(uint32_t)*mark.gpu(chunk).width,mark.height());
	DPRINT("Allocated %lu bytes for results.\n",pitch*mark.height());
	HANDLE_ERROR(cudaGetLastError()); // checking last function
	if (verbose_flag)
		d_results = (uint32_t*)malloc(sizeof(uint32_t)*mark.block_width()*mark.height());
	cudaMemset(g_results, 0, mark.height()*pitch);
	HANDLE_ERROR(cudaGetLastError()); // checking last function

	pcount += mark.gpu(chunk).width;
	startGate = ckt.size();
	const uint32_t blockcount_y = (uint32_t)(mark.gpu(chunk).width/COVER_BLOCK) + (mark.gpu(chunk).width%COVER_BLOCK > 0);
	for (uint32_t i2 = 0; i2 <= ckt.levels(); i2++) {
		const int32_t i = (ckt.levels() - (i2));
		uint32_t levelsize = ckt.levelsize(i);
		do { 
			uint32_t simblocks = min(MAX_BLOCKS, levelsize);
			dim3 numBlocks(simblocks,blockcount_y);
			startGate -= simblocks;
			assert((uint32_t)startGate + simblocks <= ckt.size());
			kernCover<N,COVER_BLOCK><<<numBlocks,COVER_BLOCK>>>(ckt.gpu_graph(), mark.gpu(chunk).data, mark.gpu(chunk).pitch,
					merges, merge_size, g_results,pitch, startGate, 
					mark.gpu(chunk).width, startPattern, ckt.offset());
			if (levelsize > MAX_BLOCKS) {
				levelsize -= simblocks;
			} else {
				levelsize = 0;
			}
		} while (levelsize > 0);
		if (i == 0) {
			// Sum for all gates and patterns
			kernSumSingle<REDUCE_THREADS><<<1,REDUCE_THREADS>>>(ckt.gpu_graph(), ckt.levelsize(0), g_results, mark.gpu(chunk).width, pitch, finalcoverage); // multithreaded, single block GPU add.
			cudaDeviceSynchronize();
			cudaMemcpy(coverage, finalcoverage, sizeof(uint64_t), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
		}
	}
	assert(startGate == 0);
	// dump to file for debugging.

	if (verbose_flag) {
		cudaMemcpy2D(d_results, sizeof(uint32_t)*mark.gpu(chunk).width, g_results, pitch, sizeof(uint32_t)*mark.gpu(chunk).width, mark.height(), cudaMemcpyDeviceToHost);
		debugCover(ckt, d_results, mark.gpu(chunk).width, mark.height(), cfile, chunk, startPattern);
		free(d_results);
	}
	cudaMemcpy(coverage, finalcoverage, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaFree(finalcoverage);
	cudaFree(g_results); // clean up.
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting


#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif

	if (verbose_flag)
		cfile.close();
#ifndef NTIMING
	return elapsed;
#else 
	return 0.0;
#endif
}
void debugCoverOutput(ARRAY2D<uint32_t> results, std::string outfile) {
#ifndef NDEBUG
	std::ofstream ofile(outfile.c_str());
		ofile << "Line:   \t";
	for ( uint32_t i = 0; i < results.height; i++) {
		ofile << std::setw(OUTJUST) << i << " ";
	}
	ofile << std::endl;
	for ( uint32_t r = 0;r < results.width; r++) {
		ofile << "Vector " << r << ":\t";
		for ( uint32_t i = 0; i < results.height; i++) {
			uint32_t z = REF2D(results.data, results.pitch, r, i);
			ofile << std::setw(OUTJUST) << (uint32_t)GET_C(z) << " "; break;
		}
		ofile << std::endl;
	}
	ofile.close();
#endif
}
#endif
