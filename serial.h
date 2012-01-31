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

float serial(Circuit& ckt, CPU_Data& input);
#endif
