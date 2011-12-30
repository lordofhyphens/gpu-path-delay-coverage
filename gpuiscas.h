#ifndef GPUISCAS_H
#define GPUISCAS_H
#include <cassert>
#include <math.h>
#include "iscas.h"
#include "defines.h"
#include "array2d.h"
#include <cuda.h>

int* gpuLoad1DVector(int* input, size_t width, size_t height);
int* loadPinned(int*, size_t);

int* gpuLoadFans(int* offset, int maxid);
void gpuShiftVectors(int* input, size_t width, size_t height);
GPUNODE* gpuLoadCircuit(const GPUNODE* graph, int maxid);
LINE* gpuLoadLines(LINE* graph, int maxid);

void freeMemory(int* data);
void freeMemory(GPUNODE* data);

ARRAY2D<int> gpuAllocateResults(size_t width, size_t height);
#endif
