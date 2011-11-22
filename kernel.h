#ifndef KERNEL_H
#define KERNEL_H

#include "iscas.h"
void runGpuSimulation(int*, int, size_t, GPUNODE*, GPUNODE*, int, LINE*, int, int*,int);
void loadLookupTables();
#endif
