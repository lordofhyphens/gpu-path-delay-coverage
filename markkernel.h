#ifndef MARKKERNEL_H
#define MARKKERNEL_H
#include <cassert>
#include <stdint.h>
#include "util/defines.h"
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/utility.h"
#include "util/segment.cuh"

float gpuMarkPaths(GPU_Data& results, GPU_Data& input, GPU_Circuit& ckt, size_t chunk, size_t startPattern);
void debugMarkOutput(ARRAY2D<uint8_t> results, const GPU_Circuit& ckt, std::string outfile = "markdebug.log");
void debugMarkOutput(GPU_Data* results, const GPU_Circuit& ckt,const size_t chunk, const size_t startPattern,std::string outfile);
void debugMarkOutput(GPU_DATA_type<coalesce_t>* results, const GPU_Circuit& ckt,const size_t chunk, const size_t startPattern,std::string outfile);
#endif
