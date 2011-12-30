#ifndef SIMKERNEL_H
#define SIMKERNEL_H
#include "iscas.h"
#include "gpuiscas.h"
float gpuRunSimulation(ARRAY2D<char> results, ARRAY2D<int> inputs, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan, int maxlevel, int pass);

void debugSimulationOutput(ARRAY2D<char> results, int pass);

#endif
