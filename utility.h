#ifndef UTILITY_H
#define UTILITY_H
#include <ctime>
#include <cmath>
#include "array2d.h"
#include <cstdlib>
	timespec diff(timespec start, timespec end);
	float floattime(timespec time);
    ARRAY2D<int> gpuAllocateBlockResults(size_t height);
#endif // UTILITY_H
