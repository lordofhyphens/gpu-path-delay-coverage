#include "coverkernel.h"
#include <cuda.h> 
void HandleCoverError( cudaError_t err, const char *file, uint32_t line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define SUM(A, B, DATA) (DATA[A]+DATA[B])

__global__ void kernSumSingle(GPUNODE* ckt, size_t size, uint32_t* input, size_t height, size_t pitch,uint64_t* meta) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < height; j++) {
			if (ckt[i].type == INPT) {
				*meta = *meta + REF2D(uint32_t, input, pitch, j, i);
			}
		}
	}
}
__device__ inline int32_t subCktFan(uint32_t* subckt, uint32_t subckt_size, uint32_t tgt) {
	// scan through the subckt list looking for tgt
	for (uint32_t i = 0; i < subckt_size; i++) { if (subckt[i] == tgt) { return i;} }
	return -1;
}
#define HANDLE_ERROR( err ) (HandleCoverError( err, __FILE__, __LINE__ ))

__global__ void kernCover(const GPUNODE* ckt, uint8_t* mark,size_t mark_pitch, int32_t* history,  uint32_t* cover, size_t cover_pitch, uint32_t* hist_cover, size_t hcover_pitch,uint32_t start_offset, uint32_t pattern_count, uint32_t start_pattern, uint32_t* offsets) { //, uint32_t* subckt, size_t subckt_size) {
    // cover is the coverage ints we're working with for this pass.
    // mark is the fresh marks
    // hist is the history of the mark status of all lines.
	const uint32_t tid = (blockIdx.y * COVER_BLOCK) + threadIdx.x;
	const uint32_t pid = tid + start_pattern; 
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
				resultCache += REF2D(uint32_t,     cover, cover_pitch, tid, FIN(offsets,gate.offset+gate.nfi,i)); // add this fanout's path count to this node.
				histCache   += REF2D(uint32_t,hist_cover,hcover_pitch, tid, FIN(offsets,gate.offset+gate.nfi,i)); // add this fanout's history path count to this node.
			}
			c = resultCache;
			h = histCache;
		}
		if (gate.type != FROM) { // FROM nodes always take the value of their fan-outs
			// c equals c+h if history[g] >= pid and line is marked
			c = (c + h)*(cache > 0)*(history[g] >= (int32_t)pid);
			// h equals 0 if history[g] >= pid, else h if this line is marked;
			h = h*(cache > 0)*(history[g] < (int32_t)pid);

        }
		// Cycle through the fanins of this node and assign them the current value
		for (uint32_t i = 0; i < gate.nfi; i++) {
			REF2D(uint32_t,cover,cover_pitch,tid, FIN(offsets,gate.offset,i)) = c; 
			REF2D(uint32_t,hist_cover,hcover_pitch,tid,FIN(offsets,gate.offset,i)) = h; 
		}
		REF2D(uint32_t, cover     , cover_pitch , tid, g) = c;
		REF2D(uint32_t, hist_cover, hcover_pitch, tid, g) = h;

//		printf("%s:%d - history[%d] = %d\n", __FILE__, __LINE__, g, history[g]);
//		printf("%s:%d - cover[%d][%d] = %d, history[%d][%d] = %d \n",__FILE__, __LINE__, tid,g, c, tid,g, h);
	}
}


float gpuCountPaths(const GPU_Circuit& ckt, GPU_Data& mark, const ARRAY2D<int32_t>& merges, uint64_t* coverage) {
	HANDLE_ERROR(cudaGetLastError()); // check to make sure there aren't any errors going into function.
	uint32_t* results, *g_results, *gh_results;
	uint64_t* finalcoverage;
	*coverage = 0;
	uint32_t startGate;
	size_t pitch, h_pitch;
	uint32_t startPattern = 0;
//	const size_t summedPatterns = (mark.width() / (MERGE_SIZE*2)) + ((mark.width() % (MERGE_SIZE*2)) > 0);

	cudaMalloc(&finalcoverage, sizeof(uint64_t));
	cudaMemset(finalcoverage, 0, sizeof(uint64_t)); // set value to 0 explicitly
	HANDLE_ERROR(cudaGetLastError()); // checking that last memory operation completed successfully.

	cudaMallocPitch(&g_results,&pitch, sizeof(uint32_t)*mark.block_width(),mark.height());
	HANDLE_ERROR(cudaGetLastError()); // checking last function
	cudaMallocPitch(&gh_results,&h_pitch,sizeof(uint32_t)*mark.block_width(),mark.height());
	HANDLE_ERROR(cudaGetLastError()); // checking last function

	cudaMemset(g_results, 0, mark.height()*pitch);
	HANDLE_ERROR(cudaGetLastError()); // checking last function
	cudaMemset(gh_results, 0, mark.height()*h_pitch);
	HANDLE_ERROR(cudaGetLastError()); // checking last function
	results = (uint32_t*)malloc(mark.block_width()*sizeof(uint32_t)*mark.height());
//	h_results = (uint32_t*)malloc(mark.block_width()*sizeof(uint32_t)*mark.height());

	ARRAY2D<uint32_t> h = ARRAY2D<uint32_t>(results, mark.height(), mark.width(), sizeof(uint32_t)*mark.width()); // on CPU 
	ARRAY2D<uint32_t> hc = ARRAY2D<uint32_t>(NULL, mark.height(), mark.width(), sizeof(uint32_t)*mark.width()); // on CPU


#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif
	uint32_t pcount = 0;
	////DPRINT("%s:%d - level count: %d\n", __FILE__,__LINE__, ckt.levels());
	for (uint32_t chunk = 0; chunk < mark.size(); chunk++) {
		pcount += mark.gpu(chunk).width;
		startGate = ckt.size();
		uint32_t blockcount_y = (uint32_t)(mark.gpu(chunk).width/COVER_BLOCK) + (mark.gpu(chunk).width%COVER_BLOCK > 0);
		//DPRINT("Patterns to process in block %u: %lu\n", chunk, mark.gpu(chunk).width);
		for (uint32_t i2 = 0; i2 < ckt.levels(); i2++) {
			int32_t i = (ckt.levels() - (i2+1));
			uint32_t levelsize = ckt.levelsize(i);
			do { 
				uint32_t simblocks = min(MAX_BLOCKS, levelsize);
				dim3 numBlocks(simblocks,blockcount_y);
				startGate -= simblocks;
	//			//DPRINT("%s:%d - running cover %d - %d\n", __FILE__,__LINE__, i, levelsize);
				assert((uint32_t)startGate < ckt.size());
				kernCover<<<numBlocks,COVER_BLOCK>>>(ckt.gpu_graph(), mark.gpu(chunk).data, mark.gpu(chunk).pitch,
						merges.data, g_results,pitch, gh_results, h_pitch, startGate, 
						pcount, startPattern, ckt.offset());
				if (levelsize > MAX_BLOCKS) {
					levelsize -= simblocks;
				} else {
					levelsize = 0;
				}
			} while (levelsize > 0);
			cudaDeviceSynchronize();
			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
		}

		kernSumSingle<<<1,1>>>(ckt.gpu_graph(), ckt.size(), g_results, h.width, pitch, finalcoverage); // inefficient singlethreaded GPU add.
		startPattern += mark.gpu(chunk).width;
	}
	cudaMemcpy(coverage, finalcoverage, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	free(results);
	cudaFree(g_results);
	cudaFree(gh_results);
	cudaFree(finalcoverage);
	#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif
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
			uint32_t z = REF2D(uint32_t, results.data, results.pitch, r, i);
			ofile << std::setw(OUTJUST) << (uint32_t)z << " "; break;
		}
		ofile << std::endl;
	}
	ofile.close();
#endif
}
