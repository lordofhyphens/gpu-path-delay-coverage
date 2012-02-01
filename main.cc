#include "utility.h"
#include "ckt.h"
#include "gpuckt.h"
#include "gpudata.h"
#include "vectors.h"
#include "simkernel.h"
#include "markkernel.h"
#include "mergekernel.h"
#include "serial.h"
#include <utility>
#include <iostream>
#include <fstream>

int main(int argc, char ** argv) {
	GPU_Circuit ckt;
	timespec start, stop;
	float elapsed = 0.0,mark,merge,cover = 0.0,sim,gpu;
	std::cerr << "Reading benchmark file " << argv[1] << "....";
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	ckt.read_bench(argv[1]);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	std::cerr << "..complete. Took " << elapsed  << "ms" << std::endl;
	std::cerr << "Copying circuit to GPU...";
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	ckt.copy(); // convert and copy circuit to GPU
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	std::cerr << "..complete. Took " << elapsed  << "ms" << std::endl;
	for (int i = 2; i < argc; i++) { // run multiple benchmark values from the same program invocation
		gpu = 0.0;
		std::cerr << "Vector set " << argv[i] << std::endl;
		std::pair<size_t,size_t> vecdim = get_vector_dim(argv[i]);
		GPU_Data *vec = new GPU_Data(vecdim.first,vecdim.second);
		std::cerr << "Reading vector file....";
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		read_vectors(*vec, argv[i], vec->pitch());
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		elapsed = floattime(diff(start, stop));
		std::cerr << "..complete. Took " << elapsed  << "ms" << std::endl;
		float serial_time = serial(ckt, *vec);
		std::cerr << "Performing serial work." << std::endl;
		std::cerr << "Serial: " << serial_time << " ms" << std::endl;

		std::cerr << "Initializing gpu memory for results...";
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		GPU_Data *sim_results = new GPU_Data(vecdim.first,ckt.size()); // initializing results array for simulation
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		elapsed = floattime(diff(start, stop));
		gpu += elapsed;

		std::cerr << "..complete." << std::endl;
		sim = gpuRunSimulation(*sim_results, *vec, ckt, 1);
		gpu += sim;
		std::cerr << "Pass 1: " << sim << " ms" << std::endl;
		
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		gpu_shift(*vec);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		elapsed = floattime(diff(start, stop));
		gpu += elapsed;

		sim = gpuRunSimulation(*sim_results, *vec, ckt, 2);
		gpu += sim;
		std::cerr << "Pass 2: " << sim << " ms" << std::endl;
		// don't need the input vectors anymore, so remove.
		delete vec;
		GPU_Data *mark_results = new GPU_Data(vecdim.first, ckt.size());
		mark = gpuMarkPaths(*mark_results, *sim_results, ckt);
		gpu += mark;
		std::cerr << "  Mark: " << mark << " ms" << std::endl;
//		std::cerr << sim_results->debug();
		delete sim_results;
		ARRAY2D<int> merge_ids = gpuAllocateBlockResults(ckt.size());
//		std::cerr << mark_results->debug();
		merge = gpuMergeHistory(*mark_results, merge_ids);  
		gpu += merge;
		std::cerr << " Merge: " << merge << " ms" << std::endl;
		delete mark_results;

		gpu += cover;

		std::cerr << "   GPU: " << elapsed << " ms" <<std::endl;
		std::cerr << "Speedup:" << serial_time/gpu << "X" <<std::endl;
		std::cout << argv[i] << ":" << vecdim.first << "," << ckt.size() <<  ";" << serial_time <<","<< gpu << "," <<  serial_time/gpu <<std::endl;
	}
	return 0;
}
