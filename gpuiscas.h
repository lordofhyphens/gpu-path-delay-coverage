#ifndef GPUISCAS_H
#define GPUISCAS_H
#include <cuda.h>
#include "iscas.h"
int* gpuLoadVectors(int** input, size_t width, size_t height);

int* gpuLoadFans(int* offset, int maxid);
GPUNODE* gpuLoadCircuit(const GPUNODE* graph, int maxid);
LINE* gpuLoadLines(LINE* graph, int maxid);

#endif
