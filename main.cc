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
#include "subckt.h"
#include <utility>
#include <iostream>
#include <fstream>

int main(int argc, char ** argv) {
	selectGPU();
	GPU_Circuit ckt;
	std::ofstream cpvec("cpuvectors.log");
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
	std::vector<SubCkt> sub_pis;
	std::vector<SubCkt> sub_pos;
	std::vector<SubCkt> sec_order;

	uint64_t *scoverage;

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
		cpvec << vec->print();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		elapsed = floattime(diff(start, stop));
		uint32_t simul_patterns = gpuCalculateSimulPatterns(ckt.size(), vecdim.first);
		std::cerr << "..complete. Took " << elapsed  << "ms" << std::endl;
		std::clog << "Maximum patterns per pass: " << simul_patterns << std::endl;
		std::clog << "Running serial simulation... " << std::endl;
		scoverage = NULL;
		float serial_time = serial(ckt, *vec, &scoverage);
//		float serial_time = 0;
		
		std::cerr << "Performing serial work." << std::endl;
		std::cerr << "Serial: " << serial_time << " ms" << std::endl;
		std::cerr << "Initializing gpu memory for results...";
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		GPU_Data *sim_results = new GPU_Data(vecdim.first,ckt.size(), simul_patterns); // initializing results array for simulation
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		elapsed = floattime(diff(start, stop));
		gpu += elapsed;

		std::cerr << "..complete." << std::endl;
		std::clog << sim_results->debug() << std::endl;
		std::clog << "Simulation pass 1...";
		sim1 = gpuRunSimulation(*sim_results, *vec, ckt, 1);
		std::clog << "..complete." << std::endl;
		gpu += sim1;
		std::cerr << "Pass 1: " << sim1 << " ms" << std::endl;
//		debugDataOutput(vec->gpu(), "siminputs.log");
//		debugSimulationOutput(sim_results->ar2d(), "gpusim-p1.log");
		// don't need the input vectors anymore, so remove.
		delete vec;
		GPU_Data *mark_results = new GPU_Data(vecdim.first,ckt.size(), simul_patterns);
		mark = gpuMarkPaths(*mark_results, *sim_results, ckt);
		gpu += mark;
		std::cerr << "  Mark: " << mark << " ms" << std::endl;
		std::cerr << sim_results->debug();
		debugMarkOutput(mark_results->ar2d(), "gpumark.log");
		delete sim_results;
		ARRAY2D<int32_t> merge_ids = gpuAllocateBlockResults(ckt.size());
		std::cerr << mark_results->debug();
		merge = gpuMergeHistory(*mark_results, merge_ids);  
		gpu += merge;
		std::cerr << " Merge: " << merge << " ms" << std::endl;
		debugMergeOutput(merge_ids, "gpumerge.log");
		
		cover = gpuCountPaths(ckt, *mark_results, merge_ids, coverage);
//		cover = 0;
		std::cerr << " Cover: " << cover << " ms" << std::endl;
		delete mark_results;
		std::cerr << "GPU Coverage: " << *coverage << std::endl;
		gpu += cover;

		std::cerr << "   GPU: " << gpu << " ms" <<std::endl;
		std::cerr << "Speedup:" << serial_time/gpu << "X" <<std::endl;

		std::cout << argv[i] << ":" << vecdim.first << "," << ckt.size() <<  ";" << serial_time <<","<< gpu << "_" << sim1 
			      <<  "_" << mark << "_"<< merge << "_" << cover << "," <<  serial_time/gpu << ":" << *scoverage << ","<< *coverage << std::endl;
		delete scoverage;
		delete coverage;
	}
	cpvec.close();
	return 0;
}
