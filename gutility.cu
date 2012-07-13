#include "utility.h"
#include "defines.h"
#include <cuda.h>
#include <stdint.h>
ARRAY2D<int32_t> gpuAllocateBlockResults(size_t height) {
	int32_t* tgt = NULL;
	cudaMalloc(&tgt, sizeof(int)*(height));
	cudaMemset(tgt, -1, sizeof(int)*height);
	return ARRAY2D<int32_t>(tgt, 1, height, sizeof(int32_t)*height);
}
void selectGPU() {
	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	cudaDeviceProp properties;
	if (num_devices > 1) {
		unsigned int max_multiprocessors = 0, max_device = 0;
		for (device = 0; device < num_devices; device++) {
			cudaGetDeviceProperties(&properties, device);
			if (max_multiprocessors < properties.totalGlobalMem) {
				max_multiprocessors = properties.totalGlobalMem;
				max_device = device;
			}
		}
		cudaSetDevice(max_device);
		cudaGetDeviceProperties(&properties, max_device);
		DPRINT("Selected %s as GPU.\n", properties.name);
	}
}

int gpuCalculateSimulPatterns(int lines, int patterns) {
	// get free memory
	size_t free_mem, total_mem;
	int allowed_patterns;
	cudaMemGetInfo(&free_mem, &total_mem);
	// added a buffer 	
	allowed_patterns = (free_mem + (lines*sizeof(int))) / (lines*(sizeof(uint32_t)*2.5) + sizeof(uint8_t)*1.5);
	return min(patterns, allowed_patterns -(allowed_patterns % 32));
}
