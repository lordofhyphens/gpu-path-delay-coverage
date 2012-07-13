#ifndef MERGEKERNEL_H
#define MERGEKERNEL_H
#include "gpuckt.h"
#include "gpudata.h"
float gpuMergeHistory(GPU_Data& input, void**);
void debugMergeOutput(size_t size, const void *, std::string outfile);
#endif
