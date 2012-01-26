#ifndef GPUISCAS_H
#define GPUISCAS_H
#include <cuda.h>
#include <cassert>
#include <math.h>
#include "iscas.h"
#include "defines.h"
#include "array2d.h"

int* gpuLoad1DVector(int* input, size_t width, size_t height);
int* loadPinned(int*, size_t);

int* gpuLoadFans(int* offset, int maxid);
void gpuShiftVectors(ARRAY2D<int> input);
GPUNODE* gpuLoadCircuit(const GPUNODE* graph, int maxid);
LINE* gpuLoadLines(LINE* graph, int maxid);

void freeMemory(int* data);
void freeMemory(char* data);
void freeMemory(GPUNODE* data);
void clearMemory(ARRAY2D<char> ar);

void gpu1PrintVectors(int* vec, size_t height, size_t width);
void gpuPrintVectors(int* vec, size_t height, size_t width);
void debugPrintVectors(ARRAY2D<int> results);
void gpuArrayCopy(ARRAY2D<char> dst, ARRAY2D<char> src);
ARRAY2D<char> gpuAllocateResults(size_t width, size_t height);
ARRAY2D<int> gpuAllocateBlockResults(size_t height);
#endif
