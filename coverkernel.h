#ifndef COVERKENREL_H
#define COVERKERNEL_H
#include "iscas.h"
#include "gpuiscas.h"

float gpuCountPaths(ARRAY2D<char> input, ARRAY2D<int> count, ARRAY2D<int> history, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan, int maxlevels);
char returnPathCount(ARRAY2D<char> results);
void debugCoverOutput(ARRAY2D<char> results);
#endif
