#ifndef COVERKENREL_H
#define COVERKERNEL_H
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/defines.h"
#include "util/utility.h"
float gpuCountPaths(const GPU_Circuit& ckt, GPU_Data& mark, const void* merge, uint64_t* coverage,  size_t chunk, size_t startPattern);
void debugCover(const Circuit& ckt, uint32_t *cover, size_t patterns, size_t lines, std::ofstream& ofile);
#endif
