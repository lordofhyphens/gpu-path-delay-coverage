#include <cuda.h>
#include <cassert>
#include "iscas.h"
#include "defines.h"
#include "markkernel.h"
void HandleMarkError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleMarkError( err, __FILE__, __LINE__ ))
#define THREAD_PER_BLOCK 64
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
__global__ void kernMerge(char* input, char* results, int offset, int width, int height, int pitch) {
	char *r;
	int result, i, dst = threadIdx.x + offset;
	if (threadIdx.x < width) {
		result = 0;
		for (i = 0; i <= blockIdx.x; i++) {
			r = ((char*)input + i*pitch);
			result = (result || r[dst]);
		}
		r = ((char*)results + pitch*blockIdx.x);
		r[dst] = result;
	}
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
		
	HANDLE_ERROR(cudaMallocArray(&cuFromProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuInptProp, &channelDesc, 4,2));
	HANDLE_ERROR(cudaMallocArray(&cuAndInptProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuAndOutpProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuOrInptProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuOrOutpProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuXorInptProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuXorOutpProp, &channelDesc, 4,4));

	HANDLE_ERROR(cudaMallocArray(&cuXorOutChain, &channelDesc, 4,2));
	HANDLE_ERROR(cudaMallocArray(&cuXorInChain, &channelDesc, 4,2));
	HANDLE_ERROR(cudaMallocArray(&cuOrInChain, &channelDesc, 4,2));
	HANDLE_ERROR(cudaMallocArray(&cuOrOutChain, &channelDesc, 4,2));
	HANDLE_ERROR(cudaMallocArray(&cuAndInChain, &channelDesc, 4,2));
	HANDLE_ERROR(cudaMallocArray(&cuAndOutChain, &channelDesc, 4,2));

	// Copying the LUTs Host->Device
	HANDLE_ERROR(cudaMemcpyToArray(cuFromProp, 0,0, from_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndInptProp, 0,0, and2_input_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndOutpProp, 0,0, and2_output_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrInptProp, 0,0, or2_input_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrOutpProp, 0,0, or2_output_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorInptProp, 0,0, xor2_input_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorOutpProp, 0,0, xor2_output_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuInptProp, 0,0, inpt_prop, sizeof(int)*8,cudaMemcpyHostToDevice));
	
	HANDLE_ERROR(cudaMemcpyToArray(cuXorInChain, 0,0, xor_inp_chain, sizeof(int)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorOutChain, 0,0, xor_outp_chain, sizeof(int)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrInChain, 0,0, or_inp_chain, sizeof(int)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrOutChain, 0,0, or_outp_chain, sizeof(int)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndInChain, 0,0, and_inp_chain, sizeof(int)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndOutChain, 0,0, and_outp_chain, sizeof(int)*8,cudaMemcpyHostToDevice));

	// Marking them as textures. LUTs should be in texture memory and cached on
	// access.
	HANDLE_ERROR(cudaBindTextureToArray(and2OutputPropLUT,cuAndOutpProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(and2InputPropLUT,cuAndInptProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(or2OutputPropLUT,cuOrOutpProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(or2InputPropLUT,cuOrInptProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xor2InputPropLUT,cuXorInptProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xor2OutputPropLUT,cuXorOutpProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(fromPropLUT,cuFromProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(inptPropLUT,cuInptProp,channelDesc));
	
	HANDLE_ERROR(cudaBindTextureToArray(XorOutChainLUT,cuXorOutChain,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(XorInChainLUT,cuXorInChain,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(OrOutChainLUT,cuOrOutChain,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(OrInChainLUT,cuOrInChain,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(AndOutChainLUT,cuAndOutChain,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(AndInChainLUT,cuAndInChain,channelDesc));
}

__global__ void kernMarkPathSegments(char *input, char* results, GPUNODE* node, int* fans, size_t width, size_t height, int ncount, int pitch) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x, nfi, goffset,val,prev;
	__shared__ int rowids[50]; // handle up to fanins of 50 
	char cache;
	int tmp = 1, pass = 0, fin1 = 0, fin2 = 0,fin = 1, type;
	char *rowResults;
	char *row;
	if (tid < height) {
		cache = 0;
		row = (char*)((char*)input + tid*pitch);
		rowResults = (char*)((char*)results + tid*pitch);
		for (int i = 0; i < width; i++) {
			rowResults[i] = 0;
		}
		for (int i = ncount-1; i >= 0; i--) {
			tmp = 1;
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
				prev = val;
			} else {
				prev = rowResults[i];
			}
			switch(type) {
				case FROM:
					// For FROM, only set the "input" line if it hasn't already
					// been set (otherwise it'll overwrite the decision of
					// another system somewhere else.

					val = (rowResults[i] > 0 && row[rowids[0]] > 1);
					rowResults[rowids[0]] = val || (rowResults[rowids[0]] > 0);
					rowResults[i] =  val;
					break;
				case BUFF:
				case NOT:
					val = tex2D(inptPropLUT, row[rowids[0]],rowResults[i]) && prev;
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
							cache = tex2D(and2OutputPropLUT, row[rowids[fin1]], row[rowids[fin2]]) && prev;
							pass += (cache > 1);
							tmp = tmp && (cache > 0);
						}
					}
					rowResults[i] = val && tmp;
					break;
				case OR:
				case NOR:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						fin = 1;
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache = tex2D(or2OutputPropLUT, row[rowids[fin1]], row[rowids[fin2]]) && prev;
							pass += (cache > 1);
							tmp = tmp && (cache > 0);
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
							cache = tex2D(xor2OutputPropLUT, row[rowids[fin1]], row[rowids[fin2]]) && prev;
							pass += (cache > 1);
							tmp = tmp && (cache > 0);
							fin = fin && tex2D(and2InputPropLUT, row[rowids[fin1]], row[rowids[fin2]]) && prev;
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
							cache = tex2D(and2InputPropLUT, row[rowids[fin1]], row[rowids[fin2]]) && prev;
							fin = cache && fin;
						}
						rowResults[rowids[fin1]] = fin;
					}
					break;
				case OR:
				case NOR:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						fin = 1;
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache = tex2D(or2InputPropLUT, row[rowids[fin1]], row[rowids[fin2]]) && prev;
							fin = cache && fin;
						}
						rowResults[rowids[fin1]] = fin;
					}
				case XOR:
				case XNOR:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						fin = 1;
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache = tex2D(xor2InputPropLUT, row[rowids[fin1]], row[rowids[fin2]]) && prev;
							fin = cache && fin;
						}
						rowResults[rowids[fin1]] = fin;
					}
					break;
				default:
					;;

			}
		}
//		// replace our working set to save memory.
//		for (int i = 0; i < width; i++) {
//			row[i] = rowResults[i];
//		}
	}
}

float gpuMarkPaths(ARRAY2D<char> input, ARRAY2D<char> results, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph,  int* fan) {
	loadPropLUTs();
	int blockcount = (int)(results.height/THREAD_PER_BLOCK) + (results.height%THREAD_PER_BLOCK > 0);
#ifndef NTIMING
	float elapsed;
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventRecord(start,0);
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	kernMarkPathSegments<<<blockcount,THREAD_PER_BLOCK>>>(input.data, results.data, dgraph.data, fan, results.width, results.height, dgraph.width, results.pitch);
	cudaDeviceSynchronize();
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&elapsed,start,stop);
//	cudaEventDestroy(start);
//	cudaEventDestroy(stop);
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}
float gpuMergeHistory(ARRAY2D<char> input, ARRAY2D<char> *mergeresult, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan) {
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
	DPRINT("height: %lu width: %lu groups: %u \n", input.height, input.width, groups);
	for (int i = 0; i < groups; i++) {
		kernMerge<<<input.height,1024,0>>>(input.data, mergeresult->data, i*1024, input.width, input.height, input.pitch);
	}
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
	cudaMemcpy(mergeresult->data, input.data, input.bwidth(), cudaMemcpyDeviceToDevice);
#ifndef NTIMING
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}

void debugMarkOutput(ARRAY2D<char> results) {
#ifndef NDEBUG
	// Routine to copy contents of our results array into host memory and print
	// it row-by-row.
	char *lvalues, *row;
	DPRINT("Post-mark results\n");
	DPRINT("Line:   \t");
	for (unsigned int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (unsigned int r = 0;r < results.height; r++) {
		lvalues = (char*)malloc(results.bwidth());
		row = ((char*)results.data + r*results.pitch); // get the current row?
		cudaMemcpy(lvalues,row,results.bwidth(),cudaMemcpyDeviceToHost);
		
		DPRINT("%s %d:\t","Vector",r);
		for (unsigned int i = 0; i < results.width; i++) {
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
	for (unsigned int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (unsigned int r = 0;r < results.height; r++) {
		lvalues = (int*)malloc(results.bwidth());
		row = (int*)((char*)results.data + r*results.pitch); // get the current row?
		cudaMemcpy(lvalues,row,results.bwidth(),cudaMemcpyDeviceToHost);
		
		DPRINT("%s %d:\t", "Vector",r);
		for (unsigned int i = 0; i < results.width; i++) {
			DPRINT("%2c ", lvalues[i] == 0 ? 'N':'S'  );
		}
		DPRINT("\n");
		free(lvalues);
	}
#endif 
}
