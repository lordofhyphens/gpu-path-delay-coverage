#include "ckt.h"
#include "gpuckt.h"
#include "gpudata.h"
#include "vectors.h"

int main(int argc, char ** argv) {
	GPU_Circuit ckt;
	GPU_Data simdata(3,5);
	ckt.read_bench(argv[1]);
	ckt.print();
	read_vectors(simdata, argv[2], 32);
	std::cout << simdata.debug();
	return 0;
}
