#ifndef UTILITY_H
#define UTILITY_H
#include <ctime>
#include <cmath>
#include "array2d.h"
#include <cstdlib>
#include <algorithm>
	timespec diff(timespec start, timespec end);
	float floattime(timespec time);
    ARRAY2D<int> gpuAllocateBlockResults(size_t height);
	void selectGPU();
	int gpuCalculateSimulPatterns(int lines, int patterns);

#endif // UTILITY_H
