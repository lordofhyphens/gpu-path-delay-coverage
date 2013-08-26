#include "gpu_hashmap.cu.h"

__device__ __host__ hashfuncs make_hashfuncs(uint32_t* g_hashlist, uint32_t* h_hashlist, int max, int slots) { hashfuncs a; a.g_hashlist = g_hashlist, a.h_hashlist = h_hashlist, a.max = max, a.slots = slots; return a;}
