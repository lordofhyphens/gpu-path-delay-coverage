#ifndef KERNEL_H
#define KERNEL_H

#include "iscas.h"
#include "gpuiscas.h"
void runGpuSimulation(ARRAY2D<int>, ARRAY2D<int> , GPUNODE*, ARRAY2D<GPUNODE>, ARRAY2D<LINE>,int*,int);
void loadLookupTables();
void loadMergeTable();
#endif
