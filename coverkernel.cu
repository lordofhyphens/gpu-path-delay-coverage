#include "coverkernel.h"
#include <cuda.h> 
void HandleCoverError( cudaError_t err, const char *file, uint32_t line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define SUM(A, B, DATA) (DATA[A]+DATA[B])
__global__ void kernSum(GPUNODE* ckt, uint32_t* input, size_t height, size_t pitch,uint32_t* meta, uint32_t mpitch, uint32_t startGate) {
	uint32_t tid = threadIdx.x;
	uint32_t gid = blockIdx.y+startGate;
	__shared__ uint32_t sdata[MERGE_SIZE];
	uint32_t i = blockIdx.x*(MERGE_SIZE*2) + threadIdx.x;
	sdata[tid] = 0;
	// need to add i and i+MERGE_SIZE for those gates whose type is INPT
	if (ckt[gid].type == INPT && i < height) {
		if (i+MERGE_SIZE > height) { // correcting for blocks smaller than MERGE_SIZE
			sdata[tid] = (REF2D(uint32_t,input,pitch,i,gid));
//			printf("%s:%d - input[%d][%d] = %d\n", __FILE__,__LINE__, i, gid, REF2D(uint32_t,input,pitch,i,gid));
		} else {
			sdata[tid] = (REF2D(uint32_t,input,pitch,i,gid)) + (REF2D(uint32_t,input,pitch,i+MERGE_SIZE,gid));
		}
		__syncthreads();

		// this is loop unrolling
		// do reduction in shared mem, comparisons against MERGE_SIZE are done at compile time.
		if (MERGE_SIZE >= 1024) { if (tid < 512 && tid+512 < height) { sdata[tid] = SUM(tid, tid+512,sdata); } __syncthreads(); }
		if (MERGE_SIZE >= 512) { if (tid < 256 && tid+256 < height) { sdata[tid] = SUM(tid, tid+256,sdata); } __syncthreads(); }
		if (MERGE_SIZE >= 256) { if (tid < 128 && tid+128 < height) { sdata[tid] = SUM(tid, tid+128,sdata); } __syncthreads(); }
		if (MERGE_SIZE >= 128) { if (tid <  64 && tid+64 < height) { sdata[tid] = SUM(tid, tid+64,sdata); } __syncthreads(); }
		if (tid < 32) {
			// Within a warp,  don't need __syncthreads();
			if (MERGE_SIZE >=  64) { if (tid+32 < height) { sdata[tid] = SUM(tid, tid + 32,sdata); } }
			if (MERGE_SIZE >=  32) { if (tid+16 < height) { sdata[tid] = SUM(tid, tid + 16,sdata); } }
			if (MERGE_SIZE >=  16) { if (tid+8 < height) { sdata[tid] = SUM(tid, tid +  8,sdata); } }
			if (MERGE_SIZE >=   8) { if (tid+4 < height) { sdata[tid] = SUM(tid, tid +  4,sdata); } }
			if (MERGE_SIZE >=   4) { if (tid+2 < height) { sdata[tid] = SUM(tid, tid +  2,sdata); } }
			if (MERGE_SIZE >=   2) { if (tid+1 < height) { sdata[tid] = SUM(tid, tid +  1,sdata); } }
		}

		// at this point, we have the sum for this gate position of the lowest. Correct by 1 to compensate for above.

		__syncthreads();
		if (threadIdx.x == 0) { REF2D(uint32_t, meta, mpitch, blockIdx.x, gid) = sdata[0]; }
		//if (threadIdx.x == 0) { printf("%s:%d - %d\n", __FILE__,__LINE__, sdata[0]); }
		__syncthreads();
	} else {
		if (threadIdx.x == 0) {REF2D(uint32_t, meta, mpitch, blockIdx.x, gid) = 0;}
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
	uint32_t tid = (blockIdx.y * COVER_BLOCK) + threadIdx.x;
	uint32_t pid = tid + start_pattern; 
//	uint32_t g = subckt[blockIdx.x+start_offset];
	uint32_t g = blockIdx.x+start_offset;
	uint32_t resultCache = 0;
	uint32_t histCache = 0;
	uint8_t cache;
	const GPUNODE& gate = ckt[g];
//	printf("%s:%d - pid = %d, pattern_count=%d\n",__FILE__,__LINE__,pid, pattern_count);
	if (pid < pattern_count) {
		cache = REF2D(uint8_t,mark,mark_pitch,tid, g); // cache the current node's marked status.
		// shorthand references to current coverage and history count.
		uint32_t c, h;

		if (gate.po == 1) {
			c = 0;
            h = (cache > 0); // set history = 1 if this line is marked.
        } else {
			c = REF2D(uint32_t, cover     , cover_pitch , tid, g);
			h = REF2D(uint32_t, hist_cover, hcover_pitch, tid, g);
		}

		if (gate.nfo > 1) {
			for (uint32_t i = 0; i < gate.nfo; i++) {
				uint32_t fot = FIN(offsets,gate.offset+gate.nfi,i); // reference to current fan-out
				resultCache += REF2D(uint32_t,cover,cover_pitch,tid,fot); // add this fanout's path count to this node.
				histCache += REF2D(uint32_t,hist_cover,hcover_pitch,tid,fot); // add this fanout's history path count to this node.
			}
			c = resultCache;
			h = histCache;
		}
		if (gate.type != FROM) {
			// needs to equal c+h if history[g] >= pid and line is marked
			c = (c + h)*(cache > 0)*(history[g] >= pid);
			// needs to equal 0 if history[g] >= pid;
			h = h*(cache > 0)*(history[g] < pid);

            for (uint32_t i = 0; i < gate.nfi; i++) {
//				uint32_t fin = subCktFan(subckt, subckt_size, FIN(offsets,gate.offset,i));
				int32_t fin = FIN(offsets,gate.offset,i);
				if (fin >= 0) {
					REF2D(uint32_t,cover,cover_pitch,tid,fin) = c; //REF2D(uint32_t,cover,cover_pitch,tid,g);
					REF2D(uint32_t,hist_cover,hcover_pitch,tid,fin) = h; //REF2D(uint32_t,hist_cover,hcover_pitch,tid,g);
				}
			}
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
	uint32_t* temp_coverage, *g_coverage;
	long* finalcoverage;
	*coverage = 0;
	uint32_t startGate;
	size_t pitch, h_pitch;
	uint32_t startPattern = 0;
	size_t summedPatterns = (mark.width() / (MERGE_SIZE*2)) + ((mark.width() % (MERGE_SIZE*2)) > 0);
//	uint32_t* debug; 
//	debug = (uint32_t*)malloc(sizeof(uint32_t)*mark.block_width()*mark.height());
	cudaMalloc(&finalcoverage, sizeof(long));
	HANDLE_ERROR(cudaGetLastError()); // checking last function
	cudaMallocPitch(&g_results,&pitch, sizeof(uint32_t)*mark.block_width(),mark.height());
	HANDLE_ERROR(cudaGetLastError()); // checking last function
	cudaMallocPitch(&gh_results,&h_pitch,sizeof(uint32_t)*mark.block_width(),mark.height());
	HANDLE_ERROR(cudaGetLastError()); // checking last function

	cudaMemset(g_results, 0, mark.height()*pitch);
	HANDLE_ERROR(cudaGetLastError()); // checking last function
	cudaMemset(gh_results, 0, mark.height()*h_pitch);
	HANDLE_ERROR(cudaGetLastError()); // checking last function
	results = (uint32_t*)malloc(mark.block_width()*sizeof(uint32_t)*mark.height());
	cudaHostAlloc(&temp_coverage, sizeof(uint32_t)*mark.height()*summedPatterns, cudaHostAllocWriteCombined | cudaHostAllocMapped);
	HANDLE_ERROR(cudaGetLastError()); // checking last function
	cudaHostGetDevicePointer(&g_coverage, temp_coverage, 0);
	HANDLE_ERROR(cudaGetLastError()); // checking last function
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
		/*
		cudaMemcpy2D(debug,sizeof(uint32_t)*mark.block_width(),g_results,pitch,sizeof(uint32_t)*mark.block_width(),mark.height(),cudaMemcpyDeviceToHost);
		for ( uint32_t r = 0;r < mark.block_width(); r++) {
			//DPRINT("Vector %d:\t",r);
			for ( uint32_t i = 0; i < mark.height(); i++) {
				uint32_t z = REF2D(uint32_t, debug, sizeof(uint32_t)*mark.block_width(), r, i);
				//DPRINT("%2d ", z);
			}
			//DPRINT("\n");
		}
*/
		size_t remaining_blocks = mark.height();
		uint32_t count = 0;
		do {
			size_t block_x = summedPatterns;//(h.width / MERGE_SIZE) + ((h.width % MERGE_SIZE) > 0);
			size_t block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
			dim3 blocks(block_x, block_y);
//			//DPRINT("%s:%d - (%lu,%lu)\n", __FILE__,__LINE__, block_x, block_y);
			kernSum<<<blocks, MERGE_SIZE>>>(ckt.gpu_graph(),g_results, h.width, pitch, g_coverage, sizeof(uint32_t)*summedPatterns, count);
			cudaDeviceSynchronize();
			count++;
			if (remaining_blocks > 65535) { remaining_blocks -= 65535; } 
			count += 65535;
			block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
		} while (remaining_blocks > 65535);
//		//DPRINT("%s:%d - summedPatterns: %lu\n", __FILE__,__LINE__, summedPatterns);
		for ( uint32_t j = 0; j < summedPatterns; j++) {
			for (uint32_t i = 0; i < ckt.size(); i++) {
//				//DPRINT("%d ", REF2D(uint32_t, temp_coverage, sizeof(uint32_t)*summedPatterns, i, j));
				*coverage = *coverage + REF2D(uint32_t, temp_coverage, sizeof(uint32_t)*summedPatterns, i, j);
			}
//			//DPRINT("\n");
		}
		startPattern += mark.gpu(chunk).width;
	}
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
