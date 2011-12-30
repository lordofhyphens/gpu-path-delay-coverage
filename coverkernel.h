#ifndef COVERKENREL_H
#define COVERKERNEL_H
#include "iscas.h"
#include "gpuiscas.h"

float gpuCountPaths(ARRAY2D<char> results, ARRAY2D<char> history, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan);
char returnPathCount(ARRAY2D<char> results);
void debugCoverOutput(ARRAY2D<char> results);
#endif
