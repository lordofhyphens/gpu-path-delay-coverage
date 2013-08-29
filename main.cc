#include "util/utility.h"
#include "util/ckt.h"
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/vectors.h"
#include "simkernel.h"
#include "markkernel.h"
#include "mergekernel.h"
#include "coverkernel.h"
#include "util/subckt.h"
#include "gpu_hashmap.cu.h"
#include <utility>
#include <iostream>
#include <fstream>
#define MAX_PATTERNS simul_patterns
using namespace std;
#undef OUTJUST
#define OUTJUST 4
int main(int argc, const char* argv[]) {
	uint8_t device = selectGPU();
	resetGPU();
	GPU_Circuit ckt;
	timespec start, stop;
	float elapsed = 0.0,mark=0.0,merge =0.0,cover = 0.0,sim1 = 0.0,gpu =0.0;
	std::cerr << "Reading benchmark file " << argv[1] << "....";
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	std::string infile(argv[1]);
	if (infile.find("bench") != std::string::npos) {
		ckt.read_bench(infile.c_str());
	} else {
			std::clog << "presorted benchmark " << infile << " ";
		ckt.load(infile.c_str());
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	std::cerr << "..complete. Took " << elapsed  << "ms" << std::endl;
//	ckt.print();
	std::cerr << "Copying circuit to GPU...";
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	ckt.copy(); // convert and copy circuit to GPU
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	std::cerr << "..complete. Took " << elapsed  << "ms" << std::endl;
	std::clog << "Circuit size is: " << ckt.size() << "Levels: " << ckt.levels() << std::endl;


	uint32_t h[HASHLENGTH];
	for (int i = 0;i < HASHLENGTH; i++) {
		h[i] = rand();
	}
	hashfuncs hashlist = make_hashfuncs(NULL,h,HASHLENGTH, HASHLENGTH);

	for (int32_t i = 2; i < argc; i++) { // run multiple benchmark values from the same program invocation
		
		uint64_t *totals = new uint64_t; 
		std::string vector_file(argv[i]);
		*totals = 0;
		gpu = 0.0;
		std::cerr << "Vector set " << vector_file << std::endl;
		std::pair<size_t,size_t> vecdim = get_vector_dim(argv[i]);
		assert(vecdim.first > 0);
		std::cerr << "Vector size: " << vecdim.first << "x"<<vecdim.second << std::endl;
		GPU_Data *vec = new GPU_Data(vecdim.first,vecdim.second, vecdim.first);
		uint32_t simul_patterns = gpuCalculateSimulPatterns(ckt.size(), vecdim.first, device);
		std::cerr << "Reading vector file....";
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		read_vectors(*vec, argv[i], vec->block_width(), vecdim.first);
		debugDataOutput(*vec, "vecout.log");
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		elapsed = floattime(diff(start, stop));
		std::cerr << "..complete. Took " << elapsed  << "ms" << std::endl;
		std::clog << "Maximum patterns per pass: " << simul_patterns << std::endl;

		std::cerr << "Initializing gpu memory for results...";
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		GPU_Data *sim_results = new GPU_Data(vecdim.first,ckt.size(), MAX_PATTERNS); // initializing results array for simulation
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		elapsed = floattime(diff(start, stop));
		gpu += elapsed;

		std::cerr << "..complete." << std::endl;
		size_t startPattern = 0;
		void* merge_ids;
		merge_ids = NULL;
		void *dc_segs = NULL;
		
		for (unsigned int chunk = 0; chunk < sim_results->size(); chunk++) {
			uint64_t *coverage = new uint64_t; 
			*coverage = 0;
			std::clog << "Simulation ...";
			sim1 = gpuRunSimulation(*sim_results, *vec, ckt, chunk, startPattern);
			std::clog << "..complete." << std::endl;
			gpu += sim1;
			std::cerr << "Simulation: " << sim1 << " ms" << std::endl;
			// don't need the input vectors anymore, so remove.
			GPU_Data *mark_results = new GPU_Data(vecdim.first,ckt.size(), MAX_PATTERNS);
			// quick test of clear code
			mark = gpuMarkPaths(*mark_results, *sim_results, ckt, chunk, startPattern);
			gpu += mark;
			std::cerr << "     Mark: " << mark << " ms" << std::endl;
			
			//merge = gpuMergeHistory(*mark_results, *sim_results, &merge_ids, chunk, startPattern);  
			merge = gpuMergeSegments(*mark_results, *sim_results, ckt, chunk, startPattern, hashlist, &dc_segs, 21);
			gpu += merge;
			std::cerr << " Merge: " << merge << " ms" << std::endl;
#ifdef LOGEXEC
			debugMergeOutput(ckt.size(), merge_ids, "gpumerge.log");
#endif //LOGEXEC
			sim_results->unload();
			cover = gpuCountPaths(ckt, *mark_results, merge_ids, coverage, chunk, startPattern);
			*totals += *coverage;
			std::cerr << " Cover: " << cover << " ms" << std::endl;
			std::cerr << "GPU Coverage: " << *coverage << ", total: "<< *totals << std::endl;
			gpu += cover;
			startPattern += mark_results->gpu(chunk).width;
			mark_results->unload();
			delete coverage;
		}
		std::cerr << "   GPU: " << gpu << " ms" <<std::endl;
		std::cout << argv[i] << ":" << vecdim.first << "," << ckt.size() <<  ";" << gpu << ";" << sim1 
			      <<  ";" << mark << ";"<< merge << ";" << cover << ";" << *totals << std::endl;
	}
	return 0;
}
