#ifndef MARKKERNEL_H
#define MARKKERNEL_H
#include <cassert>
#include <stdint.h>
#include "util/defines.h"
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/utility.h"

float gpuMarkPaths(GPU_Data& results, GPU_Data& input, GPU_Circuit& ckt);
void debugMarkOutput(ARRAY2D<uint8_t> results, std::string outfile = "markdebug.log");
void debugMarkOutput(GPU_Data* results, std::string outfile);
#endif
