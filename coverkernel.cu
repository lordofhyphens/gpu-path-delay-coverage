#include "coverkernel.h"
#include <cuda.h> 
void HandleCoverError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}


#define HANDLE_ERROR( err ) (HandleCoverError( err, __FILE__, __LINE__ ))
__global__ void kernCover(const GPUNODE* ckt, char* mark,size_t mark_pitch, int* history,  int* cover,size_t cover_pitch, int* hist_cover, size_t hcover_pitch,int start_offset, int start_pattern, int pattern_count, int* offsets) {
    // cover is the coverage ints we're working with for this pass.
    // mark is the fresh marks
    // hist is the history of the mark status of all lines.
	int tid = (blockIdx.y * COVER_BLOCK) + threadIdx.x;
	int pid = tid + start_pattern; 
	int g = blockIdx.x+start_offset;
	int resultCache = 0;
	int histCache = 0;
	char cache;
	GPUNODE gate = ckt[g];
	if (pid < pattern_count) {
		cache = REF2D(char,mark,mark_pitch,tid, g); // cache the current node's marked status.
		// shorthand references to current coverage and history count.
		int *c, *h;
		c = ADDR2D(int, cover     , cover_pitch , tid, g);
		h = ADDR2D(int, hist_cover, hcover_pitch, tid, g);

		if (gate.po == 1) {
			*c = 0;
            *h = (cache > 0); // set history = 1 if this line is marked.
        }

		if (gate.nfo > 1) {
			for (int i = 0; i < gate.nfo; i++) {
				int fot = FIN(offsets,gate.offset+gate.nfi,i); // reference to current fan-out
				resultCache += REF2D(int,cover,cover_pitch,tid,fot); // add this fanout's path count to this node.
				histCache += REF2D(int,hist_cover,hcover_pitch,tid,fot); // add this fanout's history path count to this node.
			}
			*c = resultCache;
			*h = histCache;
		}
		if (gate.type != FROM) {
//			printf("%s:%d - Before: PID: %d G %d (%d),(%d), Hist %d\n",__FILE__,__LINE__,pid, g, *c, *h, history[g]);
			*c = (*c + *h)*(cache > 0)*(history[g] >= pid);
			*h *= ((cache > 0)*(history[g] >= pid) == 0);
//			printf("%s:%d - After: PID: %d G %d (%d),(%d), Hist %d\n",__FILE__,__LINE__,pid, g, *c, *h, history[g]);

            for (int i = 0; i < gate.nfi; i++) {
				int fin = FIN(offsets,gate.offset,i);
				REF2D(int,cover,cover_pitch,tid,fin) = *c; //REF2D(int,cover,cover_pitch,tid,g);
				REF2D(int,hist_cover,hcover_pitch,tid,fin) = *h; //REF2D(int,hist_cover,hcover_pitch,tid,g);
			}
        } 
	//	if (tid == 0) { printf("%s:%d - history[%d] = %d\n",  __FILE__, __LINE__,g,history[g]);}

		//printf("%s:%d - h_results[%d][%d] = %d\n", __FILE__, __LINE__, tid, g, REF2D(int,hist_cover,hcover_pitch,tid,g));
		//printf("%s:%d - results[%d][%d] = %d\n", __FILE__, __LINE__, tid, g, *c);
	}
}


float gpuCountPaths(const GPU_Circuit& ckt, GPU_Data& mark, ARRAY2D<int> merges, int* coverage) {
	int* results, *g_results, *gh_results, *h_results;
	int startGate=ckt.size()-1;
	cudaHostAlloc(&results,sizeof(int)*mark.width()*mark.height(), cudaHostAllocMapped); // using pinned memory because we're laaaazy.
	cudaHostAlloc(&h_results,sizeof(int)*mark.width()*mark.height(), cudaHostAllocMapped);
	cudaHostGetDevicePointer(&g_results, results, 0);
	cudaHostGetDevicePointer(&gh_results, h_results, 0);
	ARRAY2D<int> h = ARRAY2D<int>(results, mark.height(), mark.width(), sizeof(int)*mark.width()); // on CPU 
	ARRAY2D<int> hc = ARRAY2D<int>(h_results, mark.height(), mark.width(), sizeof(int)*mark.width()); // on CPU


	int blockcount_y = (int)(mark.block_width()/COVER_BLOCK) + (mark.block_width()%COVER_BLOCK > 0);
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif
	for (unsigned int chunk = 0; chunk < mark.size(); chunk++) {
		for (int i = ckt.levels(); i >= 0; i--) {
			dim3 numBlocks(ckt.levelsize(i),blockcount_y);
			startGate -= (ckt.levelsize(i));
			kernCover<<<numBlocks,COVER_BLOCK>>>(ckt.gpu_graph(),mark.gpu(chunk).data,mark.gpu(chunk).pitch,merges.data,g_results,h.pitch, gh_results,hc.pitch,startGate+1, chunk*mark.block_width(),mark.gpu().width,ckt.offset());
			cudaDeviceSynchronize();
			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
		}
	}
	*coverage = 0;
	for (unsigned int j = 0; j < h.width;j++) {
		for (int i = 0; i < ckt.size(); i++) {
			if (ckt.at(i).typ == INPT) {
				*coverage = *coverage + REF2D(int,results, h.pitch, j, i);
			//	if (REF2D(int,results, h.pitch, j, i) > 0) 
			//		std::clog << "cover[" << j << "][" << i << "]: " << REF2D(int,results, h.pitch, j, i) << std::endl;
			}
		}
	}
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif
	cudaFreeHost(results);
	cudaFreeHost(h_results);
#ifndef NTIMING
	return elapsed;
#else 
	return 0.0;
#endif
}

void debugCoverOutput(ARRAY2D<int> results, std::string outfile) {
#ifndef NDEBUG
	std::ofstream ofile(outfile.c_str());
		ofile << "Line:   \t";
	for (unsigned int i = 0; i < results.height; i++) {
		ofile << std::setw(OUTJUST) << i << " ";
	}
	ofile << std::endl;
	for (unsigned int r = 0;r < results.width; r++) {
		ofile << "Vector " << r << ":\t";
		for (unsigned int i = 0; i < results.height; i++) {
			int z = REF2D(int, results.data, results.pitch, r, i);
			ofile << std::setw(OUTJUST) << (int)z << " "; break;
		}
		ofile << std::endl;
	}
	ofile.close();
#endif
}
