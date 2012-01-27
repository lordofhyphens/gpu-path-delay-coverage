#include "ckt.h"
#include "gpuckt.h"
#include "gpudata.h"
#include "vectors.h"
#include "simkernel.h"
#include <utility>
#include <iostream>
int main(int argc, char ** argv) {
	GPU_Circuit ckt;
	std::cerr << "Reading benchmark file " << argv[1] << "....";
	ckt.read_bench(argv[1]);
	std::cerr << "..complete." << std::endl;
	std::cerr << "Copying circuit to GPU...";
	ckt.copy(); // copy circuit to GPU
	std::cerr << "..complete." << std::endl;
	std::pair<size_t,size_t> vecdim = get_vector_dim(argv[2]);
	GPU_Data vec(vecdim.first,vecdim.second);
	std::cerr << "Reading vector file....";
	read_vectors(vec, argv[2], vec.pitch());
	std::cerr << "..complete." << std::endl;
	std::cerr << "Initializing gpu memory for results...";
	GPU_Data sim_results(vecdim.first,ckt.size()); // initializing results array for simulation
	std::cerr << "..complete." << std::endl;

	std::cout << gpuRunSimulation(sim_results, vec, ckt, 1) << " ms" << std::endl;
	gpu_shift(vec);
	std::cout << gpuRunSimulation(sim_results, vec, ckt, 2) << " ms" << std::endl;

//	ckt.print();
	std::cout << vec.debug() << std::endl << sim_results.debug();

	return 0;
}
