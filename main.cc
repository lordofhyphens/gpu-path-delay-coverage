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
#include <utility>
#include <iostream>
#include <fstream>
#define MAX_PATTERNS simul_patterns
#undef LOGEXEC
int main(int argc, char ** argv) {
	selectGPU();
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


	for (int32_t i = 2; i < argc; i++) { // run multiple benchmark values from the same program invocation
		uint64_t *coverage = new uint64_t; 
		gpu = 0.0;
		std::cerr << "Vector set " << argv[i] << std::endl;
		std::pair<size_t,size_t> vecdim = get_vector_dim(argv[i]);
		std::cerr << "Vector size: " << vecdim.first << "x"<<vecdim.second << std::endl;
		GPU_Data *vec = new GPU_Data(vecdim.first,vecdim.second, vecdim.first);
		std::cerr << "Reading vector file....";
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		read_vectors(*vec, argv[i], vec->block_width());
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		elapsed = floattime(diff(start, stop));
		uint32_t simul_patterns = gpuCalculateSimulPatterns(ckt.size(), vecdim.first);
		std::cerr << "..complete. Took " << elapsed  << "ms" << std::endl;
		std::clog << "Maximum patterns per pass: " << simul_patterns << std::endl;
		
		std::cerr << "Initializing gpu memory for results...";
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		GPU_Data *sim_results = new GPU_Data(vecdim.first,ckt.size(), MAX_PATTERNS); // initializing results array for simulation
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		elapsed = floattime(diff(start, stop));
		gpu += elapsed;

		std::cerr << "..complete." << std::endl;
		std::clog << "Simulation ...";
		sim1 = gpuRunSimulation(*sim_results, *vec, ckt, 1);
		std::clog << "..complete." << std::endl;
		gpu += sim1;
		std::cerr << "Simulation: " << sim1 << " ms" << std::endl;
		// don't need the input vectors anymore, so remove.
		delete vec;
		GPU_Data *mark_results = new GPU_Data(vecdim.first,ckt.size(), MAX_PATTERNS);
		mark = gpuMarkPaths(*mark_results, *sim_results, ckt);
		gpu += mark;
		std::cerr << "     Mark: " << mark << " ms" << std::endl;
		//std::cerr << sim_results->debug();
		void* merge_ids;
		//std::cerr << mark_results->debug();
		merge = gpuMergeHistory(*mark_results, *sim_results, &merge_ids);  
		gpu += merge;
		std::cerr << " Merge: " << merge << " ms" << std::endl;
#ifdef LOGEXEC
		debugMergeOutput(ckt.size(), merge_ids, "gpumerge.log");
#endif //LOGEXEC
		delete sim_results;
		cover = gpuCountPaths(ckt, *mark_results, merge_ids, coverage);

		std::cerr << " Cover: " << cover << " ms" << std::endl;
		delete mark_results;
		std::cerr << "GPU Coverage: " << *coverage << std::endl;
		gpu += cover;

		std::cerr << "   GPU: " << gpu << " ms" <<std::endl;
		std::cout << argv[i] << ":" << vecdim.first << "," << ckt.size() <<  ";" << gpu << ";" << sim1 
			      <<  ";" << mark << ";"<< merge << ";" << cover << ";" << *coverage << std::endl;
		delete coverage;
	}
	return 0;
}
