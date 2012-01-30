#ifndef SIMKERNEL_H
#define SIMKERNEL_H
#include "defines.h"
#include <cassert>
#include "gpuckt.h"
#include "gpudata.h"
#include <iomanip>
#include <ctime>
#include <string>
float gpuRunSimulation(GPU_Data& results, GPU_Data& inputs, GPU_Circuit& ckt, int pass);

void debugSimulationOutput(ARRAY2D<char> results, std::string outfile);

#endif
