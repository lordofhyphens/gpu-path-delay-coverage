#ifndef MARKKERNEL_H
#define MARKKERNEL_H
#include "iscas.h"
#include "gpuiscas.h"

float gpuMarkPaths(ARRAY2D<char> input, ARRAY2D<char> results, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph,  int* fan);

float gpuMergeHistory(ARRAY2D<char> input, ARRAY2D<char>*, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan);

void debugMarkOutput(ARRAY2D<char> results);
void debugUnionOutput(ARRAY2D<char> results);
#endif
