#ifndef MERGEKERNEL_H
#define MERGEKERNEL_H
#include "gpuckt.h"
#include "gpudata.h"
float gpuMergeHistory(GPU_Data& input, ARRAY2D<int32_t>& mergeid);
void debugMergeOutput(ARRAY2D<int32_t>& results, std::string outfile);
#endif
