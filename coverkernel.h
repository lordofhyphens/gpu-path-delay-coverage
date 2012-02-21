#ifndef COVERKENREL_H
#define COVERKERNEL_H
#include "gpuckt.h"
#include "gpudata.h"

float gpuCountPaths(const GPU_Circuit& ckt, GPU_Data& mark, ARRAY2D<int> merges,long unsigned int* coverage);
void debugCoverOutput(ARRAY2D<int> results, std::string outfile = "coveroutput.log");
#endif
