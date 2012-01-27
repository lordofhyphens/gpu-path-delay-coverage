#include "ckt.h"
#include "gpuckt.h"
#include "gpudata.h"
#include "vectors.h"
#include "simkernel.h"
#include <utility>
int main(int argc, char ** argv) {
	GPU_Circuit ckt;
	ckt.read_bench(argv[1]);
	ckt.copy(); // copy circuit to GPU
	std::pair<size_t,size_t> vecdim = get_vector_dim(argv[2]);
	GPU_Data vec(vecdim.first,vecdim.second);
	read_vectors(vec, argv[2], vec.pitch());

	GPU_Data sim_results(vecdim.first,ckt.size()); // initializing results array for simulation

	std::cout << gpuRunSimulation(sim_results, vec, ckt, 1) << " ms" << std::endl;
	gpu_shift(vec);
	std::cout << gpuRunSimulation(sim_results, vec, ckt, 2) << " ms" << std::endl;

	ckt.print();
	std::cout << vec.debug() << std::endl << sim_results.debug();

	return 0;
}
