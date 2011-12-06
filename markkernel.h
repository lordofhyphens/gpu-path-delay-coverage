#ifndef MARKKERNEL_H
#define MARKKERNEL_H
#include "iscas.h"
#include "gpuiscas.h"

float gpuMarkPaths(ARRAY2D<int> results, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph,  int* fan);

float gpuMergeHistory(ARRAY2D<int> input, int** result, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan);

void debugMarkOutput(ARRAY2D<int> results);
void debugUnionOutput(ARRAY2D<int> results);
#endif
