#include "ckt.h"
#include "gpuckt.h"

int main(int argc, char ** argv) {
	Circuit ckt;
	ckt.read_bench(argv[1]);
	ckt.print();
	return 0;
}
