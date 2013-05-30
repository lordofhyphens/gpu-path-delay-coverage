#ifndef SIMKERNEL_H
#define SIMKERNEL_H
#include <cassert>
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/defines.h"
#include "util/utility.h"
#include <iomanip>
#include <ctime>
#include <string>
#include <algorithm>
#include <stdint.h>
float gpuRunSimulation(GPU_Data& results, GPU_Data& inputs, GPU_Circuit& ckt, size_t chunk, size_t startPattern = 0);

void debugSimulationOutput(ARRAY2D<uint8_t> results, std::string outfile);
void debugSimulationOutput(GPU_Data* results, const GPU_Circuit& ckt, const size_t chunk, const size_t startPattern,std::string outfile);

#endif
