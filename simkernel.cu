#include <cuda.h>
#include "simkernel.h"
#define BLOCK_PER_KERNEL 4
#undef OUTJUST
#undef LOGEXEC
#define OUTJUST 4
// amount to unroll
#define UNROLL 4

const unsigned int SIM_BLOCK = 256;

void HandleSimError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleSimError( err, __FILE__, __LINE__ ))
texture<uint8_t, 2> and2LUT;
texture<uint8_t, 2> nand2LUT;
texture<uint8_t, 2> or2LUT;
texture<uint8_t, 2> nor2LUT;
texture<uint8_t, 2> xor2LUT;
texture<uint8_t, 2> xnor2LUT;
texture<uint8_t, 2> stableLUT;
texture<uint8_t, 1> notLUT;

 __device__ __forceinline__ uint8_t simLUT(uint8_t type, uint8_t val, uint8_t r) {
	switch(type) {
		case XOR: return tex2D(xor2LUT, val, r );
		case XNOR: return tex2D(xnor2LUT, val, r);
		case OR: return tex2D(or2LUT, val, r);
		case NOR: return tex2D(nor2LUT, val, r);
		case AND: return tex2D(and2LUT, val, r);
		case NAND: return tex2D(nand2LUT, val, r);
		default:
			return 0;
  	}
}

 __device__ __forceinline__ coalesce_t simLUT(uint8_t type, coalesce_t val, coalesce_t r) {
	coalesce_t result;
	switch(type) {
		case OR: 
			result.rows[3] = tex2D(or2LUT, val.rows[3], r.rows[3]);
			result.rows[2] = tex2D(or2LUT, val.rows[2], r.rows[2]);
			result.rows[1] = tex2D(or2LUT, val.rows[1], r.rows[1]);
			result.rows[0] = tex2D(or2LUT, val.rows[0], r.rows[0]);
			break;
		case NOR: 
			result.rows[3] = tex2D(nor2LUT, val.rows[3], r.rows[3]);
			result.rows[2] = tex2D(nor2LUT, val.rows[2], r.rows[2]);
			result.rows[1] = tex2D(nor2LUT, val.rows[1], r.rows[1]);
			result.rows[0] = tex2D(nor2LUT, val.rows[0], r.rows[0]);
			break;
		case AND: 
			result.rows[3] = tex2D(and2LUT, val.rows[3], r.rows[3]);
			result.rows[2] = tex2D(and2LUT, val.rows[2], r.rows[2]);
			result.rows[1] = tex2D(and2LUT, val.rows[1], r.rows[1]);
			result.rows[0] = tex2D(and2LUT, val.rows[0], r.rows[0]);
			break;
		case NAND:
			result.rows[3] = tex2D(nand2LUT, val.rows[3], r.rows[3]);
			result.rows[2] = tex2D(nand2LUT, val.rows[2], r.rows[2]);
			result.rows[1] = tex2D(nand2LUT, val.rows[1], r.rows[1]);
			result.rows[0] = tex2D(nand2LUT, val.rows[0], r.rows[0]);
			break;
		case XOR: 		
		case XNOR:		
		default:
			result.packed = 0;
  	}
	return result;
}
template <class T>
__launch_bounds__(SIM_BLOCK,BLOCK_PER_KERNEL) __global__ void kernSimulateP1(GPUCKT ckt, GPU_DATA_type<T> pi, GPU_DATA_type<T> output,int start_offset, int startPattern, bool last) {
	unsigned int tid = (blockIdx.y * SIM_BLOCK) + threadIdx.x;
	unsigned int gid = blockIdx.x+start_offset;
	unsigned int pid = tid + startPattern; //multiple of unrolled loop
	coalesce_t *row;
	coalesce_t val, r;
	int j;
	assert(gid < output.height);
	if (tid < output.width && pid < pi.width)  {
	//	if (tid == 0 && gid == 0) printf("Start: %d Output H,W,P: (%lu,%lu,%lu), PI H,W,P: (%lu,%lu,%lu)\n", startPattern, output.height, output.width, output.pitch, pi.height, pi.width,pi.pitch);
		const GPUNODE g = ckt.graph[gid];
		row = (coalesce_t*)((uint8_t*)output.data + gid*output.pitch)+tid; // get the line row for the current gate
		__syncthreads();
		switch (g.type) {
			case INPT:
				// Loop unroll to 4.
				// First PID 
				coalesce_t pi_p1, pi_p2;
				pi_p1 = REF2D(pi, pid, gid);
				if (pid+1 < pi.width) { // avoid a bad read
					pi_p2 = REF2D(pi,pid+1, gid); // second batch of 4
				} else {
					pi_p2 = REF2D(pi, 0, gid); // second batch of 4
				}
				val.rows[0] = tex2D(stableLUT,pi_p1.rows[0], pi_p1.rows[1]);
				val.rows[1] = tex2D(stableLUT,pi_p1.rows[1], pi_p1.rows[2]);
				val.rows[2] = tex2D(stableLUT,pi_p1.rows[2], pi_p1.rows[3]);
				val.rows[3] = tex2D(stableLUT,pi_p1.rows[3], pi_p2.rows[0]);
				break;
			case DFF:
			case BUFF:
			case FROM:
				val = REF2D(output, tid, FIN(ckt.offset, g.offset, 0));
				break;
			default: 
					// we're guaranteed at least one fanin per 
					// gate if not on an input.
					__syncthreads();
					val = REF2D(output, tid, FIN(ckt.offset, g.offset, 0));  
					if (g.type == NOT) {
						val.rows[3] = tex1D(notLUT,val.rows[3]);
						val.rows[2] = tex1D(notLUT,val.rows[2]);
						val.rows[1] = tex1D(notLUT,val.rows[1]);
						val.rows[0] = tex1D(notLUT,val.rows[0]);
					}
					j = 1;
					while (j < g.nfi) {
						__syncthreads();
						r = REF2D(output, tid, FIN(ckt.offset, g.offset, j)); 
						val = simLUT(g.type,val,r);
						j++;
					}
		}
		*row = val;
	}
/*	else {
		if (tid == 0) { printf("TID 0, pid %u, pi.width %lu", pid, pi.width); }
	}
 */
}
void loadSimLUTs() {
	uint8_t nand2[16] = {S1, S1, S1, S1, S1, S0, T1, T0, S1, T1, T1, S1, S1, T0, S1, T0};
	uint8_t and2[16]  = {S0, S0, S0, S0, S0, S1, T0, T1, S0, T0, T0, S0, S0, T1, S0, T1};
	uint8_t nor2[16]  = {S1, S0, T1, T0, S0, S0, S0, S0, T1, S0, T1, S0, T0, S0, S0, T0};
	uint8_t or2[16]   = {S0, S1, T0, T1, S1, S1, S1, S1, T0, S1, T0, S1, T1, S1, S1, T1};
	uint8_t xnor2[16] = {S1, S0, T1, T0, S0, S1, T0, T1, T1, T0, S1, S0, T0, T1, S0, S1};
	uint8_t xor2[16]  = {S0, S1, T0, T1, S1, S0, T1, T0, T0, T1, S0, S1, T1, T0, S1, S0};
	uint8_t stable[4] = {S0, T0, T1, S1};
	uint8_t not_gate[4] = {S1, S0, T1, T0};

	// device memory arrays, required. 
	cudaArray *cuNandArray, *cuAndArray,*cuNorArray, *cuOrArray,*cuXnorArray,*cuXorArray, *cuNotArray,*cuStableArray;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();

	// Allocate space in device memory for the LUTs. 
	HANDLE_ERROR(cudaMallocArray(&cuNandArray, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuAndArray, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuNorArray, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuOrArray, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuXnorArray, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuXorArray, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuStableArray, &channelDesc, 2,2));
	HANDLE_ERROR(cudaMallocArray(&cuNotArray, &channelDesc, 4,1));

	// Copying the static arrays given to device memory.
	HANDLE_ERROR(cudaMemcpyToArray(cuNandArray, 0,0, nand2, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndArray, 0,0, and2, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuNorArray, 0,0, nor2, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrArray, 0,0, or2, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXnorArray, 0,0, xnor2, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorArray, 0,0, xor2, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuStableArray, 0,0, stable, sizeof(uint8_t)*4,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuNotArray, 0,0, not_gate, sizeof(uint8_t)*4,cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaBindTextureToArray(and2LUT,cuAndArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(nand2LUT,cuNandArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(or2LUT,cuOrArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(nor2LUT,cuNorArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xor2LUT,cuXorArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xnor2LUT,cuXnorArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(stableLUT,cuStableArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(notLUT,cuNotArray,channelDesc));
}
float gpuRunSimulation(GPU_Data& results, GPU_Data& inputs, GPU_Circuit& ckt, size_t chunk, size_t initial_pattern, bool last) {
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
	loadSimLUTs(); // set up our lookup tables for simulation.
	int startGate;
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	const int blockcount_y = (int)(results.gpu(chunk).width/(SIM_BLOCK*UNROLL)) + ((results.gpu(chunk).width%(SIM_BLOCK*UNROLL)) > 0);
	startGate = 0;
	//DPRINT("Patterns to process in block %u: %lu\n", chunk, results.gpu(chunk).width);
	for (uint32_t i = 0; i <= ckt.levels(); i++) {
		int levelsize = ckt.levelsize(i);
		do { 
			int simblocks = min(MAX_BLOCKS, levelsize);
			dim3 numBlocks(simblocks,blockcount_y);
			//	DPRINT("Working on %lu patterns, %d gates in level %d\n",  results.gpu(chunk).width, simblocks, i);
			GPU_DATA_type<coalesce_t> res = toPod<coalesce_t>(results,chunk);
			GPU_DATA_type<coalesce_t> inps = toPod<coalesce_t>(inputs);
			//DPRINT("res H,W,P: %d %d %d\n", res.height, res.width, res.pitch);
			res.width = (res.width % UNROLL > 0) + (res.width / UNROLL); // set up for unroll
			inps.width = (inps.width % UNROLL > 0) + (inps.width / UNROLL); // set up for unroll
//			DPRINT("res H,W,P: %lu %lu %lu\n", res.height, res.width, res.pitch);
			kernSimulateP1<<<numBlocks,SIM_BLOCK>>>(toPod(ckt), inps, res, startGate, initial_pattern/UNROLL, last);

			startGate += simblocks;
			if (levelsize > MAX_BLOCKS) {
				levelsize -= MAX_BLOCKS;
			} else {
				levelsize = 0;
			}
		} while (levelsize > 0); 
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
	}
	cudaDeviceSynchronize();
	// We're done simulating at this point.
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif // NTIMING
#ifdef LOGEXEC
		debugSimulationOutput(&results, ckt, chunk, initial_pattern, "gpusim-p2.log");
#endif //LOGEXEC
#ifndef NTIMING
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}
void debugSimulationOutput(GPU_Data* results, const GPU_Circuit& ckt, const size_t chunk, const size_t startPattern, std::string outfile = "simdebug.log") {
#ifndef NDEBUG
	size_t t = 0;
	std::ofstream ofile;
	if (chunk == 0) {
		ofile.open(outfile.c_str(), std::ios::out);
		ofile << "Gate:     " << "\t";
		for (uint32_t i = 0; i < ckt.size(); i++) {
			ofile << std::setw(OUTJUST) << i << " ";
		}
		ofile << "\n";
	} else {
		ofile.open(outfile.c_str(), std::ios::out | std::ios::app);
	}
	DPRINT("%s, %d: Printing GPU Data chunk %lu size %lux%lu to %s.\n",__FILE__,__LINE__, chunk, results->gpu(chunk).width, results->gpu(chunk).height, outfile.c_str());
	uint8_t *lvalues;
	lvalues = (uint8_t*)malloc(results->gpu(chunk).height*results->gpu(chunk).pitch);
	cudaMemcpy2D(lvalues,results->gpu().pitch,results->gpu(chunk).data,results->gpu(chunk).pitch,results->gpu(chunk).width,results->gpu(chunk).height,cudaMemcpyDeviceToHost);
	for (unsigned int r = 0;r < results->gpu(chunk).width; r++) {
		ofile << "Vector " << t+startPattern << ":\t";
		for (unsigned int i = 0; i < results->gpu(chunk).height; i++) {
			uint8_t z = REF2D(lvalues, results->gpu(chunk).pitch, r, i);
			switch(z) {
				case S0:
					ofile  << std::setw(OUTJUST+1) << "S0 "; break;
				case S1:
					ofile  << std::setw(OUTJUST+1) << "S1 "; break;
				case T0:
					ofile  << std::setw(OUTJUST+1) << "T0 "; break;
				case T1:
					ofile  << std::setw(OUTJUST+1) << "T1 "; break;
				default:
					ofile << std::setw(OUTJUST) << (int)z << " "; break;
			}

		}
		ofile << "\n";
		t++;
	}
	free(lvalues);

	ofile.flush();
	ofile.close();
#endif

}
void debugSimulationOutput(ARRAY2D<uint8_t> results, std::string outfile = "simdebug.log") {
#ifndef NDEBUG
	uint8_t *lvalues;
	std::ofstream ofile(outfile.c_str());
	lvalues = (uint8_t*)malloc(results.height*results.pitch);
	cudaMemcpy2D(lvalues,results.pitch,results.data,results.pitch,results.width,results.height,cudaMemcpyDeviceToHost);
	for (unsigned int r = 0;r < results.width; r++) {
		ofile << "Vector " << r << ":\t";
		for (unsigned int i = 0; i < results.height; i++) {
			uint8_t z = REF2D(lvalues, results.pitch, r, i);
			switch(z) {
				case S0:
					ofile  << std::setw(OUTJUST+1) << "S0 "; break;
				case S1:
					ofile  << std::setw(OUTJUST+1) << "S1 "; break;
				case T0:
					ofile  << std::setw(OUTJUST+1) << "T0 "; break;
				case T1:
					ofile  << std::setw(OUTJUST+1) << "T1 "; break;
				default:
					ofile << std::setw(OUTJUST) << (int)z << " "; break;
			}
		}
		ofile << "\n";
	}
	free(lvalues);
	ofile.close();
#endif
}
