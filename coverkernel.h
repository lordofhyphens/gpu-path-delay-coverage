#ifndef COVERKENREL_H
#define COVERKERNEL_H
#include "iscas.h"
#include "gpuiscas.h"

float gpuCountPaths(ARRAY2D<int> results, ARRAY2D<int> history, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan);
#endif
