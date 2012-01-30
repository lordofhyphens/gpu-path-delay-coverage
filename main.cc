#include "utility.h"
#include "ckt.h"
#include "gpuckt.h"
#include "gpudata.h"
#include "vectors.h"
#include "simkernel.h"
#include <utility>
#include <iostream>
int main(int argc, char ** argv) {
	GPU_Circuit ckt;
	timespec start, stop;
	float elapsed = 0.0;
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
	std::pair<size_t,size_t> vecdim = get_vector_dim(argv[2]);
	GPU_Data *vec = new GPU_Data(vecdim.first,vecdim.second);

	std::cerr << "Reading vector file....";
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	read_vectors(*vec, argv[2], vec->pitch());
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	std::cerr << "..complete. Took " << elapsed  << "ms" << std::endl;

	std::cerr << "Initializing gpu memory for results...";
	GPU_Data *sim_results = new GPU_Data(vecdim.first,ckt.size()); // initializing results array for simulation
	std::cerr << "..complete." << std::endl;

	std::cout << "Pass 1: " << gpuRunSimulation(*sim_results, *vec, ckt, 1) << " ms" << std::endl;
	debugSimulationOutput(ARRAY2D<char>(sim_results->cpu(), sim_results->height(), sim_results->block_width()),"simdebugp1.log");
	gpu_shift(*vec);
	std::cout << "Pass 2: " << gpuRunSimulation(*sim_results, *vec, ckt, 2) << " ms" << std::endl;
	std::cout << vec->debug() << std::endl << sim_results->debug();
	// don't need the input vectors anymore, so remove.
	delete vec;

	delete sim_results;
	return 0;
}
