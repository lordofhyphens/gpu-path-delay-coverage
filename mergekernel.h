#ifndef MERGEKERNEL_H
#define MERGEKERNEL_H
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/utility.h"
float gpuMergeHistory(GPU_Data& input, GPU_Data& sim, void** mergeid, size_t chunk, uint32_t startPattern);
void debugMergeOutput(size_t size, const void *, std::string outfile);
#endif
