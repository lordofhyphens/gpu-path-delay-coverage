#ifndef MERGE_SEGMENT_CU_H
#define MERGE_SEGMENT_CU_H
/* Templated segment data structures.
 * Requires CUDA.
 *
 *
 */
#include <cuda.h>
#include <stdint.h>
#include "util/lookup.h"
// Used to store GID/PID. Since membership in the cache is sufficient, do not
// need to store anything else.


template <int N>
union hashkey {
	unsigned int k[N];
	unsigned char h[sizeof(unsigned int)*N];
};
typedef struct hashfuncs_t { uint32_t *g_hashlist, *h_hashlist, max, slots; } hashfuncs;
#if defined(__CUDACC__)
__device__ __host__ hashfuncs make_hashfuncs(uint32_t* g_hashlist, uint32_t* h_hashlist, int max, int slots);
typedef union hashable_int2_t {
	int2 key;
	unsigned char h[sizeof(int2)];
} hashable_int2;
typedef struct cache_b_t { 
	uint32_t mutex;
	hashable_int2 key;
} cache_b;
/* Generic segment structure. Uses int2 for the pid, -/+ */
template <int N>
struct segment_t { 
	uint32_t mutex;
	hashkey<N> key;
	int2 pid;
};
template <int N>
__device__ __host__ inline uint32_t getHash(const segment_t<N>& I, const uint32_t seed, uint8_t bits) {
	return hashlittle(I.key.h,sizeof(uint32_t)*N, seed) & hashmask(bits);
}
template <int N>
__device__ __host__ inline uint32_t getHash(const hashkey<N>& I, const uint32_t seed, uint8_t bits) {
	return hashlittle(I.h,sizeof(uint32_t)*N, seed) & hashmask(bits);
}
template <int N>
__device__ __host__ inline bool operator==(const hashkey<N>& lhs, const hashkey<N>&rhs) {
	bool tmp = true;
	#pragma unroll 2
	for (int i = 0; i < N; i++)
		tmp = tmp && (lhs.k[i] == rhs.k[i]);
	return tmp;
}
template <int N>
__device__ __host__ inline bool operator==(const hashkey<N>& lhs, const int &rhs) {
	bool tmp = true;
	#pragma unroll 2
	for (int i = 0; i < N; i++)
		tmp = tmp && (lhs.k[i] == rhs);
	return tmp;
}
template <int N>
__device__ __host__ inline bool operator!=(const hashkey<N>& lhs, const hashkey<N>&rhs) {
	return !(lhs == rhs);
}


template <int N> __device__ inline bool isLegal(const segment_t<N>& segment) { return (segment.key == 0); }
template <int N> __device__ inline bool isLegal(const hashkey<N>& key) { return (key == 0); }
#else
hashfuncs make_hashfuncs(uint32_t* g_hashlist, uint32_t* h_hashlist, int max, int slots);
#endif // guard against getting included for non-cuda

#endif // include guard
