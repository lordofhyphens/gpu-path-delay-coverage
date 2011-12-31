#ifndef GPUISCAS_H
#define GPUISCAS_H
#include <cassert>
#include <math.h>
#include <cuda.h>
#include "iscas.h"
#include "defines.h"
#include "array2d.h"

int* gpuLoad1DVector(int* input, size_t width, size_t height);
int* loadPinned(int*, size_t);

int* gpuLoadFans(int* offset, int maxid);
void gpuShiftVectors(int* input, size_t width, size_t height);
GPUNODE* gpuLoadCircuit(const GPUNODE* graph, int maxid);
LINE* gpuLoadLines(LINE* graph, int maxid);

void freeMemory(int* data);
void freeMemory(char* data);
void freeMemory(GPUNODE* data);
void clearMemory(ARRAY2D<char> ar);

void gpuArrayCopy(ARRAY2D<char> dst, ARRAY2D<char> src);
ARRAY2D<char> gpuAllocateResults(size_t width, size_t height);
#endif
