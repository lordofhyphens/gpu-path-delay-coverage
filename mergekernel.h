#ifndef MERGEKERNEL_H
#define MERGEKERNEL_H
#include <cuda.h>
#include "iscas.h"
#include "gpuiscas.h"

float gpuMergeHistory(ARRAY2D<char> input, ARRAY2D<char> result);

void debugUnionOutput(ARRAY2D<char> results);
#endif
