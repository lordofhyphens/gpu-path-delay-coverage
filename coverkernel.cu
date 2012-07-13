#include "coverkernel.h"
#include <cuda.h> 
void HandleCoverError( cudaError_t err, const char *file, uint32_t line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
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
#define SUM(A, B, DATA) (DATA[A]+DATA[B])
#define BLOCK_SIZE 1024
__global__ void kernSumSingle(GPUNODE* ckt, size_t size, uint32_t* input, size_t height, size_t pitch, uint64_t* meta) {
	__shared__ uint32_t sdata[BLOCK_SIZE];
	uint16_t tid = threadIdx.x;
	for (size_t i = 0; i < size; i++) { // iterate over everything in the circuit
		if (ckt[i].type != INPT) 
			continue; // short-circuit for non-PIs
		for (size_t j = 0; j < height; j+=BLOCK_SIZE) { // unrolled to handle reductions of up to BLOCK_SIZE in parallel
			sdata[tid] = 0; // reset shared data segment to 0.
			if (tid+j < height) {
				sdata[tid] = REF2D(uint32_t, input, pitch, tid+j, i);
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

__device__ inline int32_t subCktFan(uint32_t* subckt, uint32_t subckt_size, uint32_t tgt) {
	// scan through the subckt list looking for tgt
	for (uint32_t i = 0; i < subckt_size; i++) { if (subckt[i] == tgt) { return i;} }
	return -1;
}
#define HANDLE_ERROR( err ) (HandleCoverError( err, __FILE__, __LINE__ ))

__global__ void kernCover(const GPUNODE* ckt, uint8_t* mark, const size_t mark_pitch, int32_t* history,  uint32_t* cover,const size_t cover_pitch, uint32_t* hist_cover,const size_t hcover_pitch,const uint32_t start_offset,const uint32_t pattern_count,const uint32_t start_pattern, uint32_t* offsets) { //, uint32_t* subckt, size_t subckt_size) {
    // cover is the coverage ints we're working with for this pass.
    // mark is the fresh marks
    // hist is the history of the mark status of all lines.
	const uint32_t tid = (blockIdx.y * COVER_BLOCK) + threadIdx.x;
	const int32_t pid = tid + start_pattern; 
	const uint32_t g = blockIdx.x+start_offset;
	const GPUNODE& gate = ckt[g];
	
	if (pid < pattern_count) {
		const uint8_t cache = REF2D(uint8_t,mark,mark_pitch,tid, g); // cache the current node's marked status.
		// shorthand references to current coverage and history count.
		uint32_t c = 0, h = 0;

		if (gate.po == 1) {
			c = 0;
            h = (cache > 0); // set history = 1 if this line is marked.
        } else {
			c = REF2D(uint32_t, cover     , cover_pitch , tid, g);
			h = REF2D(uint32_t, hist_cover, hcover_pitch, tid, g);
		}

		if (gate.nfo > 1) { // cycle through the fanouts of this node and add their coverage values to the current node's coverage
			uint32_t resultCache = 0, histCache = 0;
			for (uint32_t i = 0; i < gate.nfo; i++) {
				resultCache += REF2D(uint32_t,     cover, cover_pitch, tid, FIN(offsets,gate.offset,gate.nfi+i)); // add this fanout's path count to this node.
				histCache   += REF2D(uint32_t,hist_cover,hcover_pitch, tid, FIN(offsets,gate.offset,gate.nfi+i)); // add this fanout's history path count to this node.
			}
			c = resultCache;
			h = histCache;
		}
		if (gate.type != FROM) { // FROM nodes always take the value of their fan-outs
			// c equals c+h if history[g] >= pid and line is marked
			c = (c + h)*(cache > 0)*(history[g] >= pid);
			// h equals 0 if history[g] >= pid, else h if this line is marked;
			h = h*(cache > 0)*(history[g] < pid);

        }
		// Cycle through the fanins of this node and assign them the current value
		for (uint32_t i = 0; i < gate.nfi; i++) {
			REF2D(uint32_t,cover,cover_pitch,tid, FIN(offsets,gate.offset,i)) = c; 
			REF2D(uint32_t,hist_cover,hcover_pitch,tid,FIN(offsets,gate.offset,i)) = h; 
		}
		REF2D(uint32_t, cover     , cover_pitch , tid, g) = c;
		REF2D(uint32_t, hist_cover, hcover_pitch, tid, g) = h;

	}
}


float gpuCountPaths(const GPU_Circuit& ckt, GPU_Data& mark, const ARRAY2D<int32_t>& merges, uint64_t* coverage) {
	HANDLE_ERROR(cudaGetLastError()); // check to make sure there aren't any errors going into function.

	std::ofstream cfile("gpucover.log", std::ios::out);
	std::ofstream chfile("gpuhcover.log", std::ios::out);
	uint32_t *g_results, *gh_results;
//	uint32_t *d_results, *dh_results; // debug results 
	uint64_t *finalcoverage;
	*coverage = 0;
	uint32_t startGate;
	size_t pitch, h_pitch;
	uint32_t startPattern = 0;
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
	for (size_t chunk = 0; chunk < mark.size(); chunk++) {
		cudaMallocPitch(&g_results,&pitch, sizeof(uint32_t)*mark.block_width(),mark.height());
		HANDLE_ERROR(cudaGetLastError()); // checking last function
		cudaMallocPitch(&gh_results,&h_pitch,sizeof(uint32_t)*mark.block_width(),mark.height());
		HANDLE_ERROR(cudaGetLastError()); // checking last function

		//	d_results = (uint32_t*)malloc(sizeof(uint32_t)*mark.block_width()*mark.height());
		//	dh_results = (uint32_t*)malloc(sizeof(uint32_t)*mark.block_width()*mark.height());
		cudaMemset(g_results, 0, mark.height()*pitch);
		HANDLE_ERROR(cudaGetLastError()); // checking last function
		cudaMemset(gh_results, 0, mark.height()*h_pitch);
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
//				std::cerr << "Working from gate " <<  startGate << " to " << startGate + simblocks << std::endl;
				kernCover<<<numBlocks,COVER_BLOCK>>>(ckt.gpu_graph(), mark.gpu(chunk).data, mark.gpu(chunk).pitch,
						merges.data, g_results,pitch, gh_results, h_pitch, startGate, 
						pcount, startPattern, ckt.offset());
				if (levelsize > MAX_BLOCKS) {
					levelsize -= simblocks;
				} else {
					levelsize = 0;
				}
				cudaDeviceSynchronize();
			} while (levelsize > 0);
			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
			if (i == 0) {
				// Sum for all gates and patterns
				kernSumSingle<<<1,BLOCK_SIZE>>>(ckt.gpu_graph(), ckt.size(), g_results, mark.gpu(chunk).width, pitch, finalcoverage); // inefficient singlethreaded GPU add.
				cudaMemcpy(coverage, finalcoverage, sizeof(uint64_t), cudaMemcpyDeviceToHost);
				std::cerr << "Current coverage: " << *coverage << std::endl;
			}
		}
		startPattern += mark.gpu(chunk).width;
		assert(startGate == 0);
		// dump to file for debugging.
//		cudaMemcpy2D(d_results, sizeof(uint32_t)*mark.gpu(chunk).width, g_results, pitch, sizeof(uint32_t)*mark.gpu(chunk).width, mark.height(), cudaMemcpyDeviceToHost);
//		cudaMemcpy2D(dh_results, sizeof(uint32_t)*mark.gpu(chunk).width, gh_results, h_pitch, sizeof(uint32_t)*mark.gpu(chunk).width, mark.height(),cudaMemcpyDeviceToHost);
//		debugCover(d_results, mark.gpu(chunk).width, mark.height(), cfile);
//		debugCover(dh_results, mark.gpu(chunk).width, mark.height(), chfile);
//		free(d_results);
//		free(dh_results);
	cudaFree(g_results); // clean up.
	cudaFree(gh_results); // clean up
	}
	cudaMemcpy(coverage, finalcoverage, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaFree(finalcoverage);
	#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif
	cfile.close();
	chfile.close();
#ifndef NTIMING
	return elapsed;
#else 
	return 0.0;
#endif
}
void debugCover(uint32_t *cover, size_t patterns, size_t lines, std::ofstream& ofile) {
#ifndef NDEBUG
	std::cerr << "Patterns: " << patterns << "; Lines: " << lines << std::endl;
	for (uint32_t r = 0; r < patterns; r++) {
		ofile << "Vector " << r << ":\t";
		for (uint32_t i = 0; i < lines; i++) {
			const uint32_t z = REF2D(uint32_t, cover, sizeof(uint32_t)*patterns, r, i);
			ofile << std::setw(OUTJUST) << z << " ";
		}
		ofile << std::endl;
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
