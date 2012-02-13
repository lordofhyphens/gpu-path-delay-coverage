#include "utility.h"
#include <cuda.h>
ARRAY2D<int> gpuAllocateBlockResults(size_t height) {
	int* tgt = NULL;
	cudaMalloc(&tgt, sizeof(int)*(height));
	cudaMemset(tgt, -1, sizeof(int)*height);
	return ARRAY2D<int>(tgt, height, 1);
}
