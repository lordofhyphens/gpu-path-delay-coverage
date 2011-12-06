#include <cuda.h>
#include "defines.h"
#include "coverkernel.h"



float gpuCountPaths(ARRAY2D<int> results, ARRAY2D<int> history, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan) {
#ifndef NTIMING
	float elapsed = 0.0;
#endif

#ifndef NTIMING
	return elapsed;
#else
	return 0.0;
#endif
}
