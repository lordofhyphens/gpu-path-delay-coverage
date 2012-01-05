#ifndef MARKKERNEL_H
#define MARKKERNEL_H
#include "defines.h"
#include <cassert>
#include <cuda.h>
#include "iscas.h"
#include "gpuiscas.h"

float gpuMarkPaths(ARRAY2D<char> input, ARRAY2D<char> results, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph,  int* fan, int);
void debugMarkOutput(ARRAY2D<char> results);
#endif
