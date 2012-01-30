#include <cuda.h>
#include "simkernel.h"

void HandleSimError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleSimError( err, __FILE__, __LINE__ ))
texture<int, 2> and2LUT;
texture<int, 2> nand2LUT;
texture<int, 2> or2LUT;
texture<int, 2> nor2LUT;
texture<int, 2> xor2LUT;
texture<int, 2> xnor2LUT;
texture<int, 2> stableLUT;
texture<int, 1> notLUT;
__global__ void kernSimulateP1(GPUNODE* graph, char* pi_data, size_t pi_pitch, size_t pi_offset, char* output_data, size_t pitch,size_t pattern_count, int* fanout_index, int start_offset) {
	int tid = (blockIdx.y * blockDim.y) + threadIdx.x;
	int gid = blockIdx.x+start_offset;
	char rowcache;
	char *row, r, val;
	int goffset, nfi, j,type;
	if (tid < pattern_count)  {
		row = ((char*)output_data + gid*pitch); // get the line row for the current gate
		goffset = graph[gid].offset;
		nfi = graph[gid].nfi;
		type = graph[gid].type;
		__syncthreads();
//		rowcache = ((char*)output_data+(fanout_index[goffset]*pitch))[tid]; 
//		if (tid == 0) printf("Block %d,G%d: Reached line %d\n", blockIdx.x, graph[gid].type,__LINE__);
		rowcache = REF2D(char,output_data,pitch,tid, FIN(fanout_index,goffset,0));
		switch (type) {
			case INPT:
				val = pi_data[gid+pi_pitch*(tid+pi_offset)];
				break;
			default: 
					// we're guaranteed at least one fanin per 
					// gate if not on an input.
					__syncthreads();
					if (type != NOT) {
						val = rowcache;
					} else {
						val = tex1D(notLUT, rowcache);
					}

					j = 1;
					while (j < nfi) {
						__syncthreads();
						r = ((char*)output_data+(fanout_index[goffset+j]*pitch))[tid];//REF2D(char,output_data,pitch,tid, FIN(fanout_index,goffset,j));  
						switch(type) {
							case XOR:
								val = tex2D(xor2LUT, val, r);break;
							case XNOR:
								val = tex2D(xnor2LUT, val, r);break;
							case OR:
								val = tex2D(or2LUT, val, r);
								break;
							case NOR:
								val = tex2D(nor2LUT, val, r);break;
							case AND:
								val = tex2D(and2LUT, val, r);break;
							case NAND:
								val = tex2D(nand2LUT, val, r);break;
						}
						j++;
					}
		}
		row[tid] = val;
	//	if (tid == 0) printf("Block %d,G%d: Reached line %d\n", blockIdx.x, gid,__LINE__);
	}
}
__global__ void kernSimulateP2(GPUNODE* graph, char* pi_data, size_t pi_pitch, size_t pi_offset, char* output_data, size_t pitch,size_t pattern_count,  int* fanout_index, int start_offset) {
	int tid = (blockIdx.y * blockDim.y) + threadIdx.x, prev=0;
	int gid = blockIdx.x+start_offset;
	__shared__ char rowcache[SIM_BLOCK];
	char *row, r;
	int goffset, nfi, val, j,type;

	if (tid < pattern_count) {
		row = ((char*)output_data + gid*pitch)+tid; // get the line row for the current gate
		goffset = graph[gid].offset;
		nfi = graph[gid].nfi;
		type = graph[gid].type;
		prev = *row;

		rowcache[threadIdx.x] = ((char*)output_data+(fanout_index[goffset]*pitch))[tid];
		switch (type) {
			case INPT:
				val = pi_data[gid+pi_pitch*(tid+pi_offset)];
				break;
			default: 
					// we're guaranteed at least one fanin per 
					// gate if not on an input.
					__syncthreads();
					if (type != NOT) {
						val = rowcache[threadIdx.x];
					} else {
						val = tex1D(notLUT, rowcache[threadIdx.x]);
					}

					j = 1;
					while (j < nfi) {
						__syncthreads();
						r = ((char*)output_data+(fanout_index[goffset+j]*pitch))[tid]; 
						switch(type) {
							case XOR:
								val = tex2D(xor2LUT, val, r );break;
							case XNOR:
								val = tex2D(xnor2LUT, val, r);break;
							case OR:
								val = tex2D(or2LUT, val, r);break;
							case NOR:
								val = tex2D(nor2LUT, val, r);break;
							case AND:
								val = tex2D(and2LUT, val, r);break;
							case NAND:
								val = tex2D(nand2LUT, val, r);break;
						}
						j++;
					}
		}
		if (type == FROM || type == BUFF)
			*row = val;
		else {
			*row = tex2D(stableLUT,prev,val);
		}
	}
}

void loadSimLUTs() {
	int nand2[16] = {1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0};
	int and2[16]  = {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1};
	int nor2[16]  = {1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0};
	int or2[16]   = {0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
	int xnor2[16] = {1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1};
	int xor2[16]  = {0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0};
	int stable[4] = {S0, T0, T1, S1};
	int not_gate[4] = {1, 1, 0, 0};

	// device memory arrays, required. 
	cudaArray *cuNandArray, *cuAndArray,*cuNorArray, *cuOrArray,*cuXnorArray,*cuXorArray, *cuNotArray,*cuStableArray;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();

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
	HANDLE_ERROR(cudaMemcpyToArray(cuNandArray, 0,0, nand2, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndArray, 0,0, and2, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuNorArray, 0,0, nor2, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrArray, 0,0, or2, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXnorArray, 0,0, xnor2, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorArray, 0,0, xor2, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuStableArray, 0,0, stable, sizeof(int)*4,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuNotArray, 0,0, not_gate, sizeof(int)*4,cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaBindTextureToArray(and2LUT,cuAndArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(nand2LUT,cuNandArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(or2LUT,cuOrArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(nor2LUT,cuNorArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xor2LUT,cuXorArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xnor2LUT,cuXnorArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(stableLUT,cuStableArray,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(notLUT,cuNotArray,channelDesc));
}

float gpuRunSimulation(GPU_Data& results, GPU_Data& inputs, GPU_Circuit& ckt, int pass = 1) {
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
	loadSimLUTs(); // set up our lookup tables for simulation.
	int startGate = 0;
	int blockcount_y = (int)(results.block_width()/SIM_BLOCK) + (results.block_width()%SIM_BLOCK > 0);
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
//	DPRINT("%s - Chunks in data result: %lu\n",__FILE__,results.size());
//	DPRINT("%s - Levels in circuit: %d\n", __FILE__, ckt.levels());
	for (unsigned int chunk = 0; chunk < results.size(); chunk++) {
		for (int i = 0; i < ckt.levels(); i++) {
			dim3 numBlocks(ckt.levelsize(i),blockcount_y);
			if (pass > 1) {
				kernSimulateP2<<<numBlocks,SIM_BLOCK>>>(ckt.gpu_graph(),inputs.gpu(),inputs.width(),chunk*results.block_width(), results.gpu(chunk), results.pitch(), inputs.block_width(), ckt.offset(), startGate);
			} else {
				kernSimulateP1<<<numBlocks,SIM_BLOCK>>>(ckt.gpu_graph(),inputs.gpu(),inputs.width(),chunk*results.block_width(), results.gpu(chunk), results.pitch(), inputs.block_width(), ckt.offset(), startGate);
			}
			startGate += ckt.levelsize(i);
			cudaDeviceSynchronize();
//					DPRINT("Pass: %d, blocks for level %d: (%d, %d) %d \n",pass, i, ckt.levelsize(i), blockcount_y, SIM_BLOCK);
			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
		}
	}
	// We're done simulating at this point.
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}

void debugSimulationOutput(ARRAY2D<char> results, std::string outfile = "simdebug.log") {
#ifndef NDEBUG
	char *lvalues, *row;
	std::ofstream ofile(outfile.c_str());
	ofile << "Post-simulation device results" << ":" << std::endl << std::endl;
	ofile << "Vector:   \t";
	ofile << std::setw(2);
	for (unsigned int i = 0; i < results.height; i++) {
		ofile << i << " ";
	}
	ofile << std::endl;
	for (unsigned int r = 0;r < results.width; r++) {
		lvalues = (char*)malloc(results.height*sizeof(char));
		row = ((char*)results.data + r*results.pitch); // get the current row?
		cudaMemcpy(lvalues,row,results.height*sizeof(char),cudaMemcpyDeviceToHost);
		ofile << "Line " << r << ":\t";
		for (unsigned int i = 0; i < results.height; i++) {
			switch(lvalues[i]) {
				case S0:
					ofile << "S0 "; break;
				case S1:
					ofile <<"S1 "; break;
				case T0:
					ofile << "T0 "; break;
				case T1:
					ofile << "T1 "; break;
				default:
					ofile << lvalues[i]; break;
			}
		}
		ofile << std::endl;
		free(lvalues);
	}
	ofile.close();
#endif
}
