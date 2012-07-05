#ifndef SERIAL_H
#define SERIAL_H
#include "defines.h"
#include "array2d.h"
#include "cpudata.h"
#include "utility.h" // some timing functions
#include "ckt.h"
#include <ctime>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <fstream>
#include <stdint.h>
float serial(Circuit& ckt, CPU_Data& input, uint64_t*);
void serial_simulate(Circuit& ckt, CPU_Data& input, const char* file);
void debugMergeOutput(int32_t* data, size_t height, size_t width, std::string outfile);
#endif
