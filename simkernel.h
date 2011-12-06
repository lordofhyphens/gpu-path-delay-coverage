#ifndef SIMKERNEL_H
#define SIMKERNEL_H
#include "iscas.h"
#include "gpuiscas.h"
float gpuRunSimulation(ARRAY2D<int> results, ARRAY2D<int> inputs, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan, int pass);

void debugSimulationOutput(ARRAY2D<int> results, int pass);

#endif
