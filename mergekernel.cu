#include "mergekernel.h"
#include <cuda.h>
#undef MERGE_SIZE
#define MERGE_SIZE 1024
void HandleMergeError( cudaError_t err, const char *file, uint32_t line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleMergeError( err, __FILE__, __LINE__ ))
#undef LOGEXEC
#define MIN(A,B,AR) ( \
		(AR[A] > 0)*(AR[A] < AR[B])*AR[A] + \
		(AR[B] > 0)*(AR[B] < AR[A])*AR[B] + \
		AR[A]*(AR[B]==0) + \
		AR[B]*(AR[A]==0) )
#undef MIN
#define MIN(A,B) ( \
		(A > 0)*(A < B)*A + \
		(B > 0)*(B < A)*B + \
		A*(B==0) + \
		B*(A==0) )
#define MIN_GEN(L,R) ( \
		((L) > 0)*((L) < (R))*(L) + \
		((R) > 0)*((R) < (L))*(R) )

__global__ void kernReduce(uint8_t* input, uint8_t* sim_input, size_t sim_pitch, size_t height, size_t pitch, int2* meta, uint32_t mpitch, uint32_t startGate, uint32_t startPattern) {
	uint32_t tid = threadIdx.x;
	uint32_t pid = tid + startPattern;
	uint32_t gid = blockIdx.y+startGate;
	int2 sdata;
	__shared__ int mx[32];
	__shared__ int my[32];
	uint8_t* row = input + pitch*gid;
	uint8_t* sim = sim_input + sim_pitch*gid;
	uint32_t i = pid;// blockIdx.x*(MERGE_SIZE*2)+tid;
	sdata = make_int2(-1,-1);
	if (tid < 32) {
		mx[tid] = -1;
		my[tid] = -1;
	}
	// Put the lower of pid and pid+MERGE_SIZE for which row[i] == 1
	// Minimum ID given by this is 1.
	if (i <= height-1) {
		int low_x = -1, low_y = -1;
		const uint8_t warp_id = threadIdx.x / (blockDim.x / 32); 
		sdata = make_int2((sim[i] == T0)*(row[i] == 1)*(i+1),(sim[i] == T1)*(row[i] == 1)*(i+1));
		__syncthreads();
		unsigned int pred_x = (sdata.x >= 1), pred_y = (sdata.y >= 1);
		low_x = __ffs(__ballot(pred_x)) - 1;
		low_y = __ffs(__ballot(pred_y)) - 1;
		mx[warp_id] = (low_x >= 0 ? low_x + (int32_t)startPattern + (warp_id * 32) : -1);
		my[warp_id] = (low_y >= 0 ? low_y + (int32_t)startPattern + (warp_id * 32) : -1);
		// at this point, we have the position of the lowest.

#ifdef LOGEXEC
		printf("%s, %d: low[%d]: (%d,%d)\n", __FILE__, __LINE__, gid, low_x, low_y);
#endif

		__syncthreads();
		if (tid < 32) {
			pred_x = mx[tid] >= 0;
			pred_y = my[tid] >= 0;
			low_x = __ffs(__ballot(pred_x)) - 1;
			low_y = __ffs(__ballot(pred_y)) - 1;
			if (threadIdx.x == 0) {
				sdata.x = (low_x >= 0 ? mx[low_x] : -1);
				sdata.y = (low_y >= 0 ? my[low_y] : -1);
#ifdef LOGEXEC
				printf("%s, %d: low[%d]: (%d,%d)\n", __FILE__, __LINE__, gid, sdata.x, sdata.y);
#endif
				REF2D(int2,meta,mpitch,blockIdx.x,blockIdx.y) = sdata; 
			}
		}

	}
}

__global__ void kernSetMin(int2* g_odata, int2* intermediate, uint32_t i_pitch, uint32_t length, uint32_t startGate, uint32_t startPattern, uint16_t chunk) {
	 uint32_t gid = blockIdx.y + startGate;
	// scan sequentially until a thread ID is discovered;
	int32_t i = 0;
	int32_t j = 0;
	while (REF2D(int2, intermediate, i_pitch, i, blockIdx.y).x < 0 && i < length && g_odata[gid].x == -1) {
		i++;
	}
	while (REF2D(int2, intermediate, i_pitch, j, blockIdx.y).y < 0 && j < length && g_odata[gid].y == -1) {
		j++;
	}
	if (i < length) {
		if (chunk > 0) {
			i = MIN_GEN(REF2D(int2, intermediate, i_pitch, i, blockIdx.y).x + startPattern, g_odata[gid].x);
		} else {
			i = REF2D(int2, intermediate, i_pitch, i, blockIdx.y).x;
		}
	} else {
		if (chunk == 0) {
			i = -1;
		} else {
			i = g_odata[gid].x;
		}
	}
#ifdef LOGEXEC
	printf("%s, %d: g_odata[%d]: (%d,%d)\n", __FILE__, __LINE__, gid,i, j);
#endif
	if (j < length) {
		if (chunk > 0) {
			j = MIN_GEN(REF2D(int2, intermediate, i_pitch, j, blockIdx.y).y + startPattern, g_odata[gid].y);
		} else {
			j = REF2D(int2, intermediate, i_pitch, j, blockIdx.y).y;
		}
	} else {
		if (chunk == 0) {
			j = -1;
		} else {
			j = g_odata[gid].y;
		}
	}

#ifdef LOGEXEC
	printf("%s, %d: g_odata[%d]: (%d,%d)\n", __FILE__, __LINE__, gid,i, j);
#endif
	g_odata[gid].x = i;
	g_odata[gid].y = j;

}
// scan through input until the first 1 is found, save the identifier and set all indicies above that.
float gpuMergeHistory(GPU_Data& input, GPU_Data& sim, void** mergeid, size_t chunk, uint32_t startPattern) {
	if (*mergeid == NULL) { cudaMalloc(mergeid, sizeof(int2)*input.height()); } // only allocate a merge table on the first pass
	int2 *mergeids = (int2*)*mergeid;
	size_t count = 0;
	int2* temparray;
	size_t pitch;

//	uint32_t* debug = (uint32_t*)malloc(sizeof(uint32_t)*input.height());
//	uint32_t* debugt = (uint32_t*)malloc(sizeof(uint32_t)*input.height()*maxblock);
//	memset(debugt, 0, input.height()*maxblock);
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	// assume that the 0th entry is the widest, which is true given the chunking method.
	cudaMallocPitch(&temparray, &pitch, sizeof(int2)*((input.gpu(0).width / MERGE_SIZE) + ((input.gpu(0).width % MERGE_SIZE) > 0)), 65535);
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting on memory allocation
#endif // NTIMING
	count = 0;
	size_t remaining_blocks = input.height();
	const size_t block_x = (input.gpu(chunk).width / MERGE_SIZE) + ((input.gpu(chunk).width % MERGE_SIZE) > 0);
	size_t block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
	do {
		dim3 blocks(block_x, block_y);
		kernReduce<<<blocks, MERGE_SIZE>>>(input.gpu(chunk).data, sim.gpu(chunk).data, sim.gpu(chunk).pitch, input.gpu(chunk).width, input.gpu(chunk).pitch, temparray, pitch, count, startPattern);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting inside the kernels
		dim3 blocksmin(1, block_y);
		kernSetMin<<<blocksmin, 1>>>(mergeids, temparray,  pitch, (block_x/2) + (block_x/2 == 0), count, startPattern, chunk);
		count+=65535;
		if (remaining_blocks < 65535) { remaining_blocks = 0;}
		if (remaining_blocks > 65535) { remaining_blocks -= 65535; }
		block_y = (remaining_blocks > 65535 ? 65535 : remaining_blocks);
	} while (remaining_blocks > 0);
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting inside the kernels
	cudaFree(temparray);
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif
#ifndef NTIMING
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}
void debugMergeOutput(size_t size, const void* res, std::string outfile) {
#ifndef NDEBUG
	int2 *lvalues, *results = (int2*)res;
	std::ofstream ofile(outfile.c_str());
	lvalues = new int2[size];
	cudaMemcpy(lvalues,results,size*sizeof(int2),cudaMemcpyDeviceToHost);
	for (size_t r = 0;r < size; r++) {
		ofile << "Gate " << r << ":\t";
		int2 z = lvalues[r];//REF2D(uint8_t, lvalues, results.pitch, r, i);
		ofile << std::setw(OUTJUST) << z.x << "," << z.y << " ";
		ofile << std::endl;
		}
	delete lvalues;
	ofile.close();
#endif
}
