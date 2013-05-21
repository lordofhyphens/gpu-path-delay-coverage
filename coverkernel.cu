#include "coverkernel.h"
#include <cuda.h>
#include "util/g_utility.h"
#undef LOGEXEC

#define BLOCK_SIZE 512
void HandleCoverError( cudaError_t err, const char *file, uint32_t line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleCoverError( err, __FILE__, __LINE__ ))
template <uint16_t blockSize>
__device__ void warpReduce(volatile uint32_t* sdata, uint16_t tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void kernSumSingle(GPUNODE* ckt, size_t size, int16_t* input, size_t height, size_t pitch, uint64_t* meta) {
	__shared__ uint32_t sdata[BLOCK_SIZE]; // only add positive #s
	uint16_t tid = threadIdx.x;
	for (size_t i = 0; i < size; i++) { // iterate over everything in the circuit
		if (ckt[i].type != INPT) 
			continue; // short-circuit for non-PIs
		for (size_t j = 0; j < height; j+=BLOCK_SIZE) { // unrolled to handle reductions of up to BLOCK_SIZE in parallel
			sdata[tid] = 0; // reset shared data segment to 0.
			if (tid+j < height) {
				sdata[tid] = REF2D(int32_t, input, pitch, tid+j, i)*(REF2D(int32_t, input, pitch, tid+j, i) >= 0); __syncthreads();
//				printf("thread %hu - sdata[%hu] = %u = %u\n", tid, tid,  REF2D(uint32_t, input, pitch, tid+j, i));
			}
			if (BLOCK_SIZE >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
			if (BLOCK_SIZE >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256];  } __syncthreads(); }
			if (BLOCK_SIZE >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128];  }  __syncthreads();}
			if (BLOCK_SIZE >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64];  } __syncthreads();}
			if (tid < 32) { warpReduce<BLOCK_SIZE>(sdata, tid); } 
			__syncthreads();
			if (threadIdx.x == 0) {
//				printf("Adding %u to total %lu for line %lu pattern %lu/%lu\n", sdata[0], *meta, i, j,height);
				*meta += sdata[0];
			}
			__syncthreads();
		}
	}
}


__global__ void kernCover(const GPUNODE* ckt, uint8_t* mark, const size_t mark_pitch, int2* history,  int16_t* cover,const size_t cover_pitch, const uint32_t start_offset,const uint32_t pattern_count,const uint32_t start_pattern, uint32_t* offsets) { //, uint32_t* subckt, size_t subckt_size) {
    // cover is the coverage ints we're working with for this pass.
	// mark is the fresh marks
	// hist is the history of the mark status of all lines.
	const uint32_t tid = (blockIdx.y * COVER_BLOCK) + threadIdx.x;
	const int32_t pid = tid + start_pattern; 
	const uint32_t g = blockIdx.x+start_offset;
	const GPUNODE& gate = ckt[g];

	if (tid < pattern_count) {
		const uint8_t cache = REF2D(uint8_t,mark,mark_pitch,tid, g); // cache the current node's marked status.
		// shorthand references to current coverage and history count.
		int32_t c = 0, h = 0;

		if (gate.po == 1) {
			c = 0;
			h = (cache > 0); // set history = 1 if this line is marked.
		} else {
			const uint8_t is_hist = REF2D(int32_t, cover, cover_pitch, tid, g) < 0;
			c = REF2D(int32_t, cover, cover_pitch, tid, g) * (1-is_hist);
			h = REF2D(int32_t, cover, cover_pitch, tid, g) * (-is_hist);
		}

		int32_t resultCache = 0, histCache = 0;
		for (uint32_t i = 0; i < gate.nfo; i++) {
			const uint32_t fot = FIN(offsets,gate.offset,gate.nfi+i);
			const uint8_t is_hist = REF2D(int32_t, cover, cover_pitch, tid, fot) < 0;
			resultCache += (REF2D(int32_t, cover, cover_pitch, tid, fot) * (1-is_hist)); // add this fanout's path count to this node.
			histCache   += (REF2D(int32_t, cover, cover_pitch, tid, fot) * (-is_hist)); // add this fanout's history path count to this node.
		}
		c += resultCache;
		h += histCache;
		assert(c >= 0);
		assert(h >= 0);
		if (gate.type != FROM) { // FROM nodes always take the value of their fan-outs
			// c equals c+h if either history[g] >= pid and line is marked
			c = (c + h)*(cache > 0)*((history[g].x == pid) + (history[g].y == pid));
			// h equals 0 if neither history[g] >= pid, else h if this line is marked;
			h = h*(cache > 0)*((history[g].x != pid)*(history[g].y != pid)) * -1;
		} else {
			h *= -1;
		}
		assert(h <= 0);
		// Cycle through the fanins of this node and assign them the current value
		REF2D(int32_t, cover     , cover_pitch , tid, g) = c + h;
		if (c > 0) { assert (h == 0); }
		if (h < 0) { assert (c == 0); }
	}
}


float gpuCountPaths(const GPU_Circuit& ckt, GPU_Data& mark, const void* merge,
		uint64_t* coverage, size_t chunk, size_t startPattern) {

	int2* merges = (int2*)merge;
	HANDLE_ERROR(cudaGetLastError()); // check to make sure there aren't any errors going into function.

	std::ofstream cfile("gpucover.log", std::ios::out);
	std::ofstream chfile("gpuhcover.log", std::ios::out);
	int16_t *g_results;
#ifdef LOGEXEC
	uint32_t *d_results, *dh_results; // debug results 
#endif //LOGEXEC
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
		DPRINT("Allocating %lu bytes for results... ",sizeof(uint16_t)*mark.gpu(chunk).width*mark.height());
		cudaMallocPitch(&g_results,&pitch, sizeof(int16_t)*mark.gpu(chunk).width,mark.height());
		DPRINT("Allocated %lu bytes for results.\n",pitch*mark.height());
		HANDLE_ERROR(cudaGetLastError()); // checking last function
#ifdef LOGEXEC
		d_results = (uint16_t*)malloc(sizeof(uint16_t)*mark.block_width()*mark.height());
		dh_results = (uint16_t*)malloc(sizeof(uint16_t)*mark.block_width()*mark.height());
#endif // LOGEXEC
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
				kernCover<<<numBlocks,COVER_BLOCK>>>(ckt.gpu_graph(), mark.gpu(chunk).data, mark.gpu(chunk).pitch,
						merges, g_results,pitch, startGate, 
						mark.gpu(chunk).width, startPattern, ckt.offset());
				if (levelsize > MAX_BLOCKS) {
					levelsize -= simblocks;
				} else {
					levelsize = 0;
				}
			} while (levelsize > 0);
			if (i == 0) {
				// Sum for all gates and patterns
				kernSumSingle<<<1,BLOCK_SIZE>>>(ckt.gpu_graph(), ckt.levelsize(0), g_results, mark.gpu(chunk).width, pitch, finalcoverage); // multithreaded, single block GPU add.
				cudaDeviceSynchronize();
				cudaMemcpy(coverage, finalcoverage, sizeof(uint64_t), cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
				HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
			}
		}
		startPattern += mark.gpu(chunk).width;
		assert(startGate == 0);
		// dump to file for debugging.

#ifdef LOGEXEC
		cudaMemcpy2D(d_results, sizeof(uint16_t)*mark.gpu(chunk).width, g_results, pitch, sizeof(uint16_t)*mark.gpu(chunk).width, mark.height(), cudaMemcpyDeviceToHost);
		cudaMemcpy2D(dh_results, sizeof(uint16_t)*mark.gpu(chunk).width, gh_results, h_pitch, sizeof(uint16_t)*mark.gpu(chunk).width, mark.height(),cudaMemcpyDeviceToHost);
		debugCover(ckt, d_results, mark.gpu(chunk).width, mark.height(), cfile);
		debugCover(ckt, dh_results, mark.gpu(chunk).width, mark.height(), chfile);
		free(d_results);
		free(dh_results);
#endif // LOGEXEC
		cudaFree(g_results); // clean up.
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
	cudaMemcpy(coverage, finalcoverage, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaFree(finalcoverage);
	#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif

#ifdef LOGEXEC
	cfile.close();
	chfile.close();
#endif
#ifndef NTIMING
	return elapsed;
#else 
	return 0.0;
#endif
}
void debugCover(const Circuit& ckt, uint16_t *cover, size_t patterns, size_t lines, std::ofstream& ofile) {
#ifndef NDEBUG
	std::cerr << "Patterns: " << patterns << "; Lines: " << lines << std::endl;
	for (uint32_t r = 0; r < patterns; r++) {
		ofile << "Vector " << r << ":\t";
		for (uint32_t i = 0; i < lines; i++) {
			if (ckt.at(i).typ == INPT) {
				const uint16_t z = REF2D(uint16_t, cover, sizeof(uint16_t)*patterns, r, i);
				ofile << std::setw(OUTJUST) << z << " ";
			}
		}
		ofile << "\n";
	}
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
			uint32_t z = REF2D(uint32_t, results.data, results.pitch, r, i);
			ofile << std::setw(OUTJUST) << (uint32_t)z << " "; break;
		}
		ofile << std::endl;
	}
	ofile.close();
#endif
}
