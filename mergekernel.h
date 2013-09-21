#ifndef MERGEKERNEL_H
#define MERGEKERNEL_H
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/utility.h"
float gpuMergeHistory(GPU_Data& input, GPU_Data& sim, void** mergeid, size_t chunk, uint32_t startPattern);
float gpuMergeSegments(GPU_Data& mark, GPU_Data& sim, GPU_Circuit& ckt, size_t chunk, uint32_t ext_startPattern, void** seglist);
void debugMergeOutput(size_t size, const void *, std::string outfile);
#endif
