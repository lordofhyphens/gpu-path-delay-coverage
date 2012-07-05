#ifndef MARKKERNEL_H
#define MARKKERNEL_H
#include "defines.h"
#include <cassert>
#include "gpuckt.h"
#include "gpudata.h"

float gpuMarkPaths(GPU_Data& results, GPU_Data& input, GPU_Circuit& ckt);
void debugMarkOutput(ARRAY2D<uint8_t> results, std::string outfile = "markdebug.log");
#endif
