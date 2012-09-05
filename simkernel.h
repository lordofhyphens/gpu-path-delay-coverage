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
float gpuRunSimulation(GPU_Data& results, GPU_Data& inputs, GPU_Circuit& ckt, uint8_t pass);

void debugSimulationOutput(ARRAY2D<uint8_t> results, std::string outfile);
void debugSimulationOutput(GPU_Data* results, std::string outfile);

#endif
