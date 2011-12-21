#ifndef SERIAL_H
#define SERIAL_H
#include "defines.h"
#include "iscas.h"
#include "gpuiscas.h"
void LOGIC_gate(int i, GPUNODE* node, int* fans, int* res, size_t height, size_t width , int pass);
float cpuRunSimulation(ARRAY2D<int> results, ARRAY2D<int> inputs, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan, int pass);
void INPT_gate(int i, int pi, ARRAY2D<int> results, ARRAY2D<int> input, GPUNODE* graph, int* fans,int pass);

void cpuShiftVectors(int* input, size_t width, size_t height);
int* cpuLoadFans(int* offset, int maxid);
LINE* cpuLoadLines(LINE* graph, int maxid);
GPUNODE* cpuLoadCircuit(const GPUNODE* graph, int maxid);
int* cpuLoadVectors(int** input, size_t width, size_t height);
int* cpuLoad1DVector(int* input, size_t width, size_t height);
float cpuCountPaths(ARRAY2D<int> results, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan, int*);

float cpuMarkPaths(ARRAY2D<int> results, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph,  int* fan);
float cpuMergeHistory(ARRAY2D<int> input, int** mergeresult, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan);

int sReturnPathCount(ARRAY2D<int> results);
void debugCpuMark(ARRAY2D<int> results);
void debugCpuSimulationOutput(ARRAY2D<int> results, int pass);
#endif
