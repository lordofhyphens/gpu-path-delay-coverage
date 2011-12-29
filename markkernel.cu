#include <cuda.h>
#include <cassert>
#include "iscas.h"
#include "defines.h"
#include "markkernel.h"

#define THREAD_PER_BLOCK 256
texture<int, 2> and2OutputPropLUT;
texture<int, 2> and2InputPropLUT;
texture<int, 2> or2OutputPropLUT;
texture<int, 2> or2InputPropLUT;
texture<int, 2> xor2OutputPropLUT;
texture<int, 2> xor2InputPropLUT;
texture<int, 2> fromPropLUT;
texture<int, 2> inptPropLUT;
texture<int, 2> mergeLUT;

texture<int, 2> AndInChainLUT;
texture<int, 2> AndOutChainLUT;
texture<int, 2> OrInChainLUT;
texture<int, 2> OrOutChainLUT;
texture<int, 2> XorInChainLUT;
texture<int, 2> XorOutChainLUT;

// group all results together, this implementation will fail if # of lines > 1024
// will need to group lines into groups of 1024 or less
__global__ void kernMerge(int* input, int* results, int offset, int width, int height) {
	int *r,result, i, dst = threadIdx.x + offset;
	if (threadIdx.x < width) {
		result = 0;
		for (i = 0; i <= blockIdx.x; i++) {
			r = (int*)((char*)input + sizeof(int)*i*width);
			result = (result || r[dst]);
		}
		r = (int*)((char*)results + sizeof(int)*width*blockIdx.x);
		r[dst] = result;
	}
}
void loadMergeLUTs() {
	int merge[4] = {0,1,1,1};
	cudaArray *cuMerge;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();	
	cudaMallocArray(&cuMerge,&channelDesc, 2, 2);
	cudaMemcpyToArray(cuMerge, 0,0, merge, sizeof(int)*4,cudaMemcpyHostToDevice);
	cudaBindTextureToArray(mergeLUT,cuMerge,channelDesc);
}
void loadPropLUTs() {
	// Creating a set of static arrays that represent our LUTs
		// Addressing for the propagations:
	// 2 4x4 groups such that 
	int and2_output_prop[16]= {0,0,0,0,0,2,1,1,0,1,1,0,0,1,1,1};
	int and2_input_prop[16] = {0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1};
	int or2_output_prop[16] = {2,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1};
	int or2_input_prop[16]  = {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	int xor2_input_prop[16] = {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	int xor2_output_prop[16]= {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	int from_prop[16]       = {0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1};
	int inpt_prop[8]        = {0,0,0,0,0,0,1,1};

	int and_outp_chain[8]   = {0,0,1,0,0,1,1,1};
	int and_inp_chain[8]    = {0,0,0,0,0,1,1,1};
	int or_outp_chain[8]    = {0,0,0,1,1,0,1,1};
	int or_inp_chain[8]     = {0,0,0,0,1,0,1,1};
	int xor_outp_chain[8]   = {0,0,0,0,0,0,0,0};
	int xor_inp_chain[8]    = {0,0,0,0,0,0,0,0};

	cudaExtent volumeSize = make_cudaExtent(4,4,2);
	// device memory arrays, required. 
	cudaArray *cuAndInptProp, *cuAndOutpProp, *cuOrInptProp, *cuOrOutpProp, *cuFromProp, *cuInptProp, *cuXorInptProp, *cuXorOutpProp;
	cudaArray *cuAndOutChain, *cuAndInChain, *cuOrInChain, *cuOrOutChain, *cuXorInChain, *cuXorOutChain;
	// generic formatting information. All of our arrays are the same, so sharing it shouldn't be a problem.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
	// Allocating memory on the device.
		
	cudaMallocArray(&cuFromProp, &channelDesc, 4,4);
	cudaMallocArray(&cuInptProp, &channelDesc, 4,2);
	cudaMallocArray(&cuAndInptProp, &channelDesc, 4,4);
	cudaMallocArray(&cuAndOutpProp, &channelDesc, 4,4);
	cudaMallocArray(&cuOrInptProp, &channelDesc, 4,4);
	cudaMallocArray(&cuOrOutpProp, &channelDesc, 4,4);
	cudaMallocArray(&cuXorInptProp, &channelDesc, 4,4);
	cudaMallocArray(&cuXorOutpProp, &channelDesc, 4,4);

	cudaMallocArray(&cuXorOutChain, &channelDesc, 4,2);
	cudaMallocArray(&cuXorInChain, &channelDesc, 4,2);
	cudaMallocArray(&cuOrInChain, &channelDesc, 4,2);
	cudaMallocArray(&cuOrOutChain, &channelDesc, 4,2);
	cudaMallocArray(&cuAndInChain, &channelDesc, 4,2);
	cudaMallocArray(&cuAndOutChain, &channelDesc, 4,2);

	// Copying the LUTs Host->Device
	cudaMemcpyToArray(cuFromProp, 0,0, from_prop, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuAndInptProp, 0,0, and2_input_prop, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuAndOutpProp, 0,0, and2_output_prop, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuOrInptProp, 0,0, or2_input_prop, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuOrOutpProp, 0,0, or2_output_prop, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXorInptProp, 0,0, xor2_input_prop, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXorOutpProp, 0,0, xor2_output_prop, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuInptProp, 0,0, inpt_prop, sizeof(int)*8,cudaMemcpyHostToDevice);
	
	cudaMemcpyToArray(cuXorInChain, 0,0, xor_inp_chain, sizeof(int)*8,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXorOutChain, 0,0, xor_outp_chain, sizeof(int)*8,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuOrInChain, 0,0, or_inp_chain, sizeof(int)*8,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuOrOutChain, 0,0, or_outp_chain, sizeof(int)*8,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuAndInChain, 0,0, and_inp_chain, sizeof(int)*8,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuAndOutChain, 0,0, and_outp_chain, sizeof(int)*8,cudaMemcpyHostToDevice);

	// Marking them as textures. LUTs should be in texture memory and cached on
	// access.
	cudaBindTextureToArray(and2OutputPropLUT,cuAndOutpProp,channelDesc);
	cudaBindTextureToArray(and2InputPropLUT,cuAndInptProp,channelDesc);
	cudaBindTextureToArray(or2OutputPropLUT,cuOrOutpProp,channelDesc);
	cudaBindTextureToArray(or2InputPropLUT,cuOrInptProp,channelDesc);
	cudaBindTextureToArray(xor2InputPropLUT,cuXorInptProp,channelDesc);
	cudaBindTextureToArray(xor2OutputPropLUT,cuXorOutpProp,channelDesc);
	cudaBindTextureToArray(fromPropLUT,cuFromProp,channelDesc);
	cudaBindTextureToArray(inptPropLUT,cuInptProp,channelDesc);
	
	cudaBindTextureToArray(XorOutChainLUT,cuXorOutChain,channelDesc);
	cudaBindTextureToArray(XorInChainLUT,cuXorInChain,channelDesc);
	cudaBindTextureToArray(OrOutChainLUT,cuOrOutChain,channelDesc);
	cudaBindTextureToArray(OrInChainLUT,cuOrInChain,channelDesc);
	cudaBindTextureToArray(AndOutChainLUT,cuAndOutChain,channelDesc);
	cudaBindTextureToArray(AndInChainLUT,cuAndInChain,channelDesc);
}

__global__ void kernMarkPathSegments(int *results, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x, nfi, goffset,val;
	__shared__ int rowids[50]; // handle up to fanins of 50 
	__shared__ char cache[THREAD_PER_BLOCK]; // needs to be 2x # of threads being run
	int tmp = 1, pass = 0, fin1 = 0, fin2 = 0,fin = 1, type;
	int *rowResults, *row;
	if (tid < height) {
		cache[threadIdx.x] = 0;
		row = (int*)((char*)results + tid*(width)*sizeof(int));
		rowResults = (int*)malloc(sizeof(int)*width);
		for (int i = 0; i < ncount; i++) {
			rowResults[i] = 0;
		}
		for (int i = ncount-1; i >= 0; i--) {
			nfi = node[i].nfi;
			type = node[i].type;
			if (threadIdx.x == 0) {
				goffset = node[i].offset;
				// preload all of the fanin line #s for this gate to shared memory.
				for (int j = 0; j < nfi;j++) {
					rowids[j] = fans[goffset+j];
				}
			}
			__syncthreads();
			// switching based on value causes divergence, switch based on node type.
			val = (row[i] > 1);
			if (node[i].po) {
				rowResults[i] = val;
			}
			switch(type) {
				case FROM:
					// For FROM, only set the "input" line if it hasn't already
					// been set (otherwise it'll overwrite the decision of
					// another system somewhere else.

					val = (rowResults[i] > 0 && row[rowids[0]] > 1);
					rowResults[rowids[0]] = val || rowResults[i];
					rowResults[i] =  val;
					break;
				case BUFF:
				case NOT:
					val = tex2D(inptPropLUT, row[rowids[0]],rowResults[i]) && rowResults[i];
					rowResults[rowids[0]] = val;
					rowResults[i] = val;
					break;
					// For the standard gates, setting three values -- both the
					// input lines and the output line.  row[i]-1 is the
					// transition on the output, offset to make the texture
					// calculations correct because there are 4 possible values
					// row[i] can take: 0, 1, 2, 3.  0, 1 are the same, as are
					// 2,3, so we subtract 1 and clamp to an edge if we
					// overflow.
					// 0 becomes -1 (which clamps to 0)
					// 1 becomes 0
					// 2 becomes 1
					// 3 becomes 2 (which clamps to 1)
					// There's only 2 LUTs for each gate type. The input LUT
					// checks for path existance through the first input, so we
					// call it twice with the inputs reversed to check both
					// paths.

				case NAND:
				case AND:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache[threadIdx.x] = tex2D(and2OutputPropLUT, row[rowids[fin1]], row[rowids[fin2]]);
							pass += (cache[threadIdx.x] > 1);
							tmp = tmp && ((int)cache[threadIdx.x] > 0);
						}
					}
					rowResults[i] = val && tmp && (pass <= nfi);
					break;
				case OR:
				case NOR:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						fin = 1;
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache[threadIdx.x] = tex2D(or2OutputPropLUT, row[rowids[fin1]], row[rowids[fin2]]);
							pass += (cache[threadIdx.x] > 1);
							tmp = tmp && ((int)cache[threadIdx.x] > 0);
						}
					}
					rowResults[i] = val && tmp && (pass <= nfi);
					break;
				case XOR:
				case XNOR:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						fin = 1;
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache[threadIdx.x] = tex2D(xor2OutputPropLUT, row[rowids[fin1]], row[rowids[fin2]]);
							pass += (cache[threadIdx.x] > 1);
							tmp = tmp && ((int)cache[threadIdx.x] > 0);
							fin = fin && tex2D(and2InputPropLUT, row[rowids[fin1]], row[rowids[fin2]]);
						}
					}
					rowResults[i] = val && tmp && (pass <= nfi);
					break;
				default:
					// if there is a transition that will propagate, set = to some positive #?
					break;
			}
			switch(type) {
				case AND:
				case NAND:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						fin = 1;
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache[threadIdx.x] = tex2D(and2InputPropLUT, row[rowids[fin1]], row[rowids[fin2]]);
							fin = cache[threadIdx.x] && fin;
						}
						rowResults[rowids[fin1]] = fin && rowResults[i];
					}
					break;
				case OR:
				case NOR:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						fin = 1;
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache[threadIdx.x] = tex2D(or2InputPropLUT, row[rowids[fin1]], row[rowids[fin2]]);
							fin = cache[threadIdx.x] && fin;
						}
						rowResults[rowids[fin1]] = fin && rowResults[i];
					}
				case XOR:
				case XNOR:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						fin = 1;
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache[threadIdx.x] = tex2D(xor2InputPropLUT, row[rowids[fin1]], row[rowids[fin2]]);
							fin = cache[threadIdx.x] && fin;
						}
						rowResults[rowids[fin1]] = fin && rowResults[i];
					}
					break;
				default:
					;;

			}
		}
		// replace our working set to save memory.
		for (int i = 0; i < width; i++) {
			row[i] = rowResults[i];
		}
		free(rowResults);
	}
}

float gpuMarkPaths(ARRAY2D<int> results, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph,  int* fan) {
	loadPropLUTs();
#ifndef NTIMING
	float elapsed;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
#endif // NTIMING
	int blockcount = (int)(results.height/THREAD_PER_BLOCK) + (results.height%THREAD_PER_BLOCK > 0);
	kernMarkPathSegments<<<blockcount,THREAD_PER_BLOCK>>>(results.data, dgraph.data, fan, results.width, results.height, dgraph.width);
	cudaDeviceSynchronize();
#ifndef NTIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}
float gpuMergeHistory(ARRAY2D<int> input, int** mergeresult, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan) {
//	loadMergeLUTs();
	cudaMalloc(mergeresult, sizeof(int)*input.height*input.width);
	// for bigger circuits or more patterns, need some logic here to divide work according to what will fit. 
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventRecord(start,0);
#endif // NTIMING
	int groups = (input.width / 1024) + 1; 
	cudaStream_t *streams;
	streams = new cudaStream_t[groups];
	DPRINT("height: %d width: %d groups: %d \n", input.height, input.width, groups);
	for (int i = 0; i < groups; i++) {
		cudaStreamCreate(streams+i);
		kernMerge<<<input.height,1024,0,streams[i]>>>(input.data, *mergeresult, i*1024, input.width, input.height);
	}
	for (int i =0; i < groups; i++)
		cudaStreamSynchronize(streams[i]);
	cudaDeviceSynchronize();
#ifndef NTIMING
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&elapsed,start,stop);
//	cudaEventDestroy(start);
//	cudaEventDestroy(stop);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif
	for (int i = 0; i < groups;i++) {
		cudaStreamDestroy(streams[i]);
	}
	cudaMemcpy(*mergeresult, input.data, input.bwidth(), cudaMemcpyDeviceToDevice);
#ifndef NTIMING
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}

void debugMarkOutput(ARRAY2D<int> results) {
#ifndef NDEBUG
	// Routine to copy contents of our results array into host memory and print
	// it row-by-row.
	int *lvalues, *row;
	DPRINT("Post-mark results\n");
	DPRINT("Line:   \t");
	for (int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (int r = 0;r < results.height; r++) {
		lvalues = (int*)malloc(results.bwidth());
		row = (int*)((char*)results.data + r*results.bwidth()); // get the current row?
		cudaMemcpy(lvalues,row,results.bwidth(),cudaMemcpyDeviceToHost);
		
		DPRINT("%s %d:\t","Vector",r);
		for (int i = 0; i < results.width; i++) {
			DPRINT("%2d ", lvalues[i]);//== 0 ? 'N':'S'  );
		}
		DPRINT("\n");
		free(lvalues);
	}
#endif 
}
void debugUnionOutput(ARRAY2D<int> results) {
#ifndef NDEBUG
	// Routine to copy contents of our results array into host memory and print
	// it row-by-row.
	int *lvalues, *row;
	DPRINT("Post-union results\n");
	DPRINT("Line:   \t");
	for (int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (int r = 0;r < results.height; r++) {
		lvalues = (int*)malloc(results.bwidth());
		row = (int*)((char*)results.data + r*results.bwidth()); // get the current row?
		cudaMemcpy(lvalues,row,results.bwidth(),cudaMemcpyDeviceToHost);
		
		DPRINT("%s %d:\t", "Vector",r);
		for (int i = 0; i < results.width; i++) {
			DPRINT("%2c ", lvalues[i] == 0 ? 'N':'S'  );
		}
		DPRINT("\n");
		free(lvalues);
	}
#endif 
}
