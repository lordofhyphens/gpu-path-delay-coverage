#include "utility.h"
#include "ckt.h"
#include "gpuckt.h"
#include "gpudata.h"
#include "vectors.h"
#include "simkernel.h"
#include "markkernel.h"
#include "mergekernel.h"
#include "coverkernel.h"
#include "serial.h"
#include <utility>
#include <iostream>
#include <fstream>

int main(int argc, char ** argv) {
	GPU_Circuit ckt;
	std::ofstream cpvec("cpuvectors.log");
	timespec start, stop;
	float elapsed = 0.0,mark,merge,cover = 0.0,sim,gpu;
	std::cerr << "Reading benchmark file " << argv[1] << "....";
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	ckt.read_bench(argv[1]);
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
	for (int i = 2; i < argc; i++) { // run multiple benchmark values from the same program invocation
		gpu = 0.0;
		std::cerr << "Vector set " << argv[i] << std::endl;
		std::pair<size_t,size_t> vecdim = get_vector_dim(argv[i]);
		std::cerr << "Vector size: " << vecdim.first << "x"<<vecdim.second << std::endl;
		GPU_Data *vec = new GPU_Data(vecdim.first,vecdim.second);
		std::cerr << "Reading vector file....";
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		read_vectors(*vec, argv[i], vec->block_width());
		cpvec << vec->print();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		elapsed = floattime(diff(start, stop));
		std::cerr << "..complete. Took " << elapsed  << "ms" << std::endl;
		float serial_time = serial(ckt, *vec);
		std::cerr << "Performing serial work." << std::endl;
		std::cerr << "Serial: " << serial_time << " ms" << std::endl;
		std::cerr << vec->debug();
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
		//debugDataOutput(vec->gpu(), "siminputs.log");
		debugSimulationOutput(sim_results->ar2d(), "gpusim-p1.log");
// Don't need this routine if I just shift tids by 1 in the second sim pass.
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		gpu_shift(*vec);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		elapsed = floattime(diff(start, stop));
		gpu += elapsed;

		sim = gpuRunSimulation(*sim_results, *vec, ckt, 2);
		gpu += sim;
		std::cerr << "Pass 2: " << sim << " ms" << std::endl;
		debugSimulationOutput(sim_results->ar2d(), "gpusim-p2.log");
		//debugDataOutput(vec->gpu(), "siminputs-shifted.log");
		// don't need the input vectors anymore, so remove.
		delete vec;
		GPU_Data *mark_results = new GPU_Data(vecdim.first, ckt.size());
		mark = gpuMarkPaths(*mark_results, *sim_results, ckt);
		gpu += mark;
		std::cerr << "  Mark: " << mark << " ms" << std::endl;
		std::cerr << sim_results->debug();
		debugMarkOutput(mark_results->ar2d(), "gpumark.log");
		delete sim_results;
		ARRAY2D<int> merge_ids = gpuAllocateBlockResults(ckt.size());
		std::cerr << mark_results->debug();
		merge = gpuMergeHistory(*mark_results, merge_ids);  
		gpu += merge;
		std::cerr << " Merge: " << merge << " ms" << std::endl;
		int *coverage = new int;
		cover = gpuCountPaths(ckt, *mark_results, merge_ids, coverage);
		debugCoverOutput(ARRAY2D<int>(coverage,mark_results->height(), mark_results->width(),mark_results->width()*sizeof(int)));
		std::cerr << " Cover: " << cover << " ms" << std::endl;
		delete mark_results;
		std::cerr << "GPU Coverage: " << *coverage << std::endl;
		gpu += cover;

		std::cerr << "   GPU: " << elapsed << " ms" <<std::endl;
		std::cerr << "Speedup:" << serial_time/gpu << "X" <<std::endl;
		std::cout << argv[i] << ":" << vecdim.first << "," << ckt.size() <<  ";" << serial_time <<","<< gpu << "," <<  serial_time/gpu <<std::endl;
		delete coverage;
	}
	cpvec.close();
	return 0;
}
