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
#define THREAD_PER_BLOCK 32
#define MTHREAD_BLOCK 32
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
texture<char, 2> inputTexture;
// group all results together, this implementation will fail if # of lines > 1024
// will need to group lines into groups of 1024 or less
__global__ void kernMerge(char* input, char* results,int width, int height, int pitch, int rpitch) {
	char *mrow, *irow;
	__shared__ char current[MTHREAD_BLOCK];
	int tid = (blockDim.y * blockIdx.y) + threadIdx.x;
	if (tid < width) {
		results[tid] = input[tid];
		irow = input+(blockIdx.x*pitch);
		mrow = results+(blockIdx.x*rpitch);
		current[threadIdx.x] = irow[0];
		for (unsigned int i = 1; i < tid; i++){
			current[threadIdx.x] = irow[i] || current[threadIdx.x];
		}
		mrow[tid] = current[threadIdx.x];
	}
}

/*
   	Parallel reduction
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)
	Modified from Cuda SDK to fit our specific problem (not sum, but logical OR, which is close enough.)
*/
template <unsigned int blockSize>
__global__ void reduce6(char *g_idata, char *g_odata, size_t n, size_t pitch)
{
    extern __shared__ int sdata[];
	char* irow = g_idata+(blockIdx.x*pitch);
	char* mrow = g_odata+(blockIdx.x*pitch);

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.y*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.y;
    sdata[tid] = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        sdata[tid] |= irow[i] || irow[i+blockSize];  
        i += gridSize;
    } 
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] |= sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] |= sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] |= sdata[tid +  64]; } __syncthreads(); }
    
	// unroll at compile time! 
    {
        if (blockSize >=  64) { sdata[tid] |= sdata[tid + 32];  }
        if (blockSize >=  32) { sdata[tid] |= sdata[tid + 16];  }
        if (blockSize >=  16) { sdata[tid] |= sdata[tid +  8];  }
        if (blockSize >=   8) { sdata[tid] |= sdata[tid +  4];  }
        if (blockSize >=   4) { sdata[tid] |= sdata[tid +  2];  }
        if (blockSize >=   2) { sdata[tid] |= sdata[tid +  1];  }
    }
    
    // write result for this block to global mem 
    if (tid == 0) mrow[blockIdx.x] = sdata[0];
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

__global__ void kernMarkPathSegments(char *input, char* results, GPUNODE* node, int* fans, size_t width, size_t height, int start, int pitch) {
	int tid = (blockIdx.y * blockDim.y) + threadIdx.x, nfi, goffset,val,prev;
	int gid = (blockIdx.x) + start;
	__shared__ char rowCache[THREAD_PER_BLOCK][100];
	__shared__ char resultCache[THREAD_PER_BLOCK][100];
	char cache;
	int tmp = 1, pass = 0, fin1 = 0, fin2 = 0,fin = 1, type,g;
	char *rowResults;
	char *row;
	if (tid < height) {
		cache = 0;
		row = (char*)((char*)input + gid*pitch);
		rowResults = (char*)((char*)results + gid*pitch);
		tmp = 1;
		nfi = node[gid].nfi;
		type = node[gid].type;
		goffset = node[gid].offset;
		rowCache[threadIdx.x][0] = row[tid];
		resultCache[threadIdx.x][0] = rowResults[tid];
		for (int q = 0; q < nfi; q++) {
//			printf("T: %d BID: %d G: %d Placing %d in gate position %d\n", tid, blockIdx.x, gid, ((char*)input+(fans[goffset+q]*pitch))[tid], fans[goffset+q]);
			rowCache[threadIdx.x][q+1] = ((char*)input+(fans[goffset+q]*pitch))[tid];
			resultCache[threadIdx.x][q+1] = ((char*)results+(fans[goffset+q]*pitch))[tid];
		}
		__syncthreads();
		// switching based on value causes divergence, switch based on node type.
		val = (rowCache[threadIdx.x][0] > 1);
		if (node[gid].po) {
			resultCache[threadIdx.x][0] = val;
			prev = val;
		} else {
			prev = resultCache[threadIdx.x][0];
		}
		switch(type) {
			case FROM:
				// For FROM, only set the "input" line if it hasn't already
				// been set (otherwise it'll overwrite the decision of
				// another system somewhere else.
				val = (resultCache[threadIdx.x][0] > 0 && (rowCache[threadIdx.x][0] > 1));
				g = val || (resultCache[threadIdx.x][1] > 0);
				resultCache[threadIdx.x][1] |= g;
				resultCache[threadIdx.x][0] = val;
				break;
			case BUFF:
			case NOT:
				val = tex2D(inptPropLUT, rowCache[threadIdx.x][0],resultCache[threadIdx.x][0]) && prev;
				resultCache[threadIdx.x][1] = val;
				resultCache[threadIdx.x][0] = val;
				break;
				// For the standard gates, setting three values -- both the
				// input lines and the output line.  rowCache[threadIdx.x][i]-1 is the
				// transition on the output, offset to make the texture
				// calculations correct because there are 4 possible values
				// rowCache[threadIdx.x][i] can take: 0, 1, 2, 3.  0, 1 are the same, as are
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
				for (fin1 = 1; fin1 <= nfi; fin1++) {
					for (fin2 = 1; fin2 <= nfi; fin2++) {
						if (fin1 == fin2) continue;
						cache = tex2D(and2OutputPropLUT, rowCache[threadIdx.x][fin1], rowCache[threadIdx.x][fin2]);
						pass += (cache > 1);
						tmp = tmp && (cache > 0);
					}
				}
				resultCache[threadIdx.x][0] = val && tmp && (pass <= nfi) && prev;
//				printf("T: %d BID: %d G: %d PASS: %d, TMP: %d, PREV: %d, VAL: %d\n", tid, blockIdx.x, gid, pass, tmp, prev, resultCache[threadIdx.x][0]);
				break;
			case OR:
			case NOR:
				for (fin1 = 0; fin1 < nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < nfi; fin2++) {
						if (fin1 == fin2) continue;
						cache = tex2D(or2OutputPropLUT, rowCache[threadIdx.x][fin1], rowCache[threadIdx.x][fin2]);
						pass += (cache > 1);
						tmp = tmp && (cache > 0);
					}
				}
				resultCache[threadIdx.x][0] = val && tmp && (pass <= nfi) && prev;
				break;
			case XOR:
			case XNOR:
				for (fin1 = 0; fin1 < nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < nfi; fin2++) {
						if (fin1 == fin2) continue;
						cache = tex2D(xor2OutputPropLUT, rowCache[threadIdx.x][fin1], rowCache[threadIdx.x][fin2]);
						pass += (cache > 1);
						tmp = tmp && (cache > 0);
					}
				}
				resultCache[threadIdx.x][0] = val && tmp && (pass <= nfi) && prev;
				break;
			default:
				// if there is a transition that will propagate, set = to some positive #?
				break;
		}
		for (fin1 = 0; fin1 < nfi; fin1++) {
			if (nfi < 2) continue;
			fin = 1;
			for (fin2 = 0; fin2 < nfi; fin2++) {
				if (fin1 == fin2) continue;
				switch(type) {
					case AND:
					case NAND:
						cache = tex2D(and2InputPropLUT, rowCache[threadIdx.x][fin1+1], rowCache[threadIdx.x][fin2+1]); break;
					case OR:
					case NOR:
						cache = tex2D(or2InputPropLUT, rowCache[threadIdx.x][fin1+1], rowCache[threadIdx.x][fin2+1]); break;
					case XOR:
					case XNOR:
						cache = tex2D(xor2InputPropLUT, rowCache[threadIdx.x][fin1+1], rowCache[threadIdx.x][fin2+1]); break;
				}
				fin = cache && fin && prev;
			}
			resultCache[threadIdx.x][fin1+1] = fin;
		}
		// stick the contents of resultCache into the results array
		__syncthreads();

//		printf("T: %d BID: %d G: %d PASS: %d, TMP: %d, PREV: %d, VAL: %d\n", tid, blockIdx.x, gid, pass, tmp, prev, resultCache[threadIdx.x][0]);
		rowResults[tid] = resultCache[threadIdx.x][0];
		for (int j = 0; j < nfi; j++) {
			((char*)results+(fans[goffset+j]*pitch))[tid] = resultCache[threadIdx.x][j+1];
		}
	}
}

float gpuMarkPaths(ARRAY2D<char> input, ARRAY2D<char> results, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph,  int* fan, int maxlevels) {
	loadPropLUTs();
	int *gatesinLevel, startGate=0;
	gatesinLevel = new int[maxlevels];
	for (int i = 0; i < maxlevels; i++) {
		gatesinLevel[i] = 0;
		for (unsigned int j = 0; j < results.width; j++) {
			if (graph[j].level == i) {
				gatesinLevel[i]++;
			}
		}
		startGate += gatesinLevel[i];
	}
	int blockcount_y = (int)(results.height/THREAD_PER_BLOCK) + (results.height%THREAD_PER_BLOCK > 0);

#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	for (int i = maxlevels-1; i >= 0; i--) {
		dim3 numBlocks(gatesinLevel[i],blockcount_y);
		startGate -= gatesinLevel[i];
//		DPRINT("Starting gate: %d, level %d\n", startGate, i);
		kernMarkPathSegments<<<numBlocks,THREAD_PER_BLOCK>>>(input.data, results.data, dgraph.data, fan, results.width, results.height, startGate, results.pitch);
		cudaDeviceSynchronize();
	}
	free(gatesinLevel);
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}
float gpuMergeHistory(ARRAY2D<char> input, ARRAY2D<char> *mergeresult, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan) {
//	int blockcount = (input.height / MTHREAD_BLOCK) + 1;
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	
	dim3 blocks(input.width, input.height);
	//kernMerge<<<blocks,MTHREAD_BLOCK>>>(input.data, mergeresult->data, input.width, input.height, input.pitch, mergeresult->pitch);
	reduce6<MTHREAD_BLOCK><<< blocks, MTHREAD_BLOCK, MTHREAD_BLOCK*sizeof(int) >>>(input.data, mergeresult->data, input.height, input.pitch);

	cudaDeviceSynchronize();
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
#endif
//	cudaMemcpy2D(mergeresult->data, mergeresult->pitch, input.data, input.pitch, input.width, input.height, cudaMemcpyDeviceToDevice);
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
	DPRINT("Vector:   \t");
	for (unsigned int i = 0; i < results.height; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (unsigned int r = 0;r < results.width; r++) {
		lvalues = (char*)malloc(results.height*sizeof(char));
		row = ((char*)results.data + r*results.pitch); // get the current row?
		cudaMemcpy(lvalues,row,results.height*sizeof(char),cudaMemcpyDeviceToHost);
		
		DPRINT("%s\t%d:\t","Line",r);
		for (unsigned int i = 0; i < results.height; i++) {
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
