#ifndef COVERKENREL_H
#define COVERKERNEL_H
#include "gpuckt.h"
#include "gpudata.h"
#include "defines.h"
float gpuCountPaths(const GPU_Circuit& ckt, GPU_Data& mark, const ARRAY2D<int32_t>& merges, uint64_t* coverage);
void debugCoverOutput(ARRAY2D<uint32_t> results, std::string outfile = "coveroutput.log");
#endif
