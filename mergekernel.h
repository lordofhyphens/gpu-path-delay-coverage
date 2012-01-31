#ifndef MERGEKERNEL_H
#define MERGEKERNEL_H
#include "gpuckt.h"
#include "gpudata.h"
float gpuMergeHistory(GPU_Data& input, ARRAY2D<int> mergeid);
void debugUnionOutput(ARRAY2D<char> results);
#endif
