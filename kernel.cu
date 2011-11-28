#include <cuda.h>
#include <cassert>
#include "iscas.h"
#include "kernel.h"
#include "defines.h"

texture<int, 2> and2LUT;
texture<int, 2> nand2LUT;
texture<int, 2> or2LUT;
texture<int, 2> nor2LUT;
texture<int, 2> xor2LUT;
texture<int, 2> xnor2LUT;
texture<int, 2> stableLUT;
__global__ void XOR_gate(int i, int* fans, GPUNODE* graph, int *res, int PATTERNS,size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int val;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[graph[i].offset]];
		while (j < graph[i].nfi) {
			val = tex2D(xor2LUT, val, row[fans[graph[i].offset+j]]);
			j++;
		}
		row[fans[graph[i].offset+graph[i].nfi]] = val;
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}
__global__ void XNOR_gate(int i, int* fans, GPUNODE* graph, int *res, int PATTERNS,size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int val;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[graph[i].offset]];
		while (j < graph[i].nfi) {
			val = tex2D(xnor2LUT, val, row[fans[graph[i].offset+j]]);
			j++;
		}
		row[fans[graph[i].offset+graph[i].nfi]] = val;
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}
__global__ void OR_gate(int i, int* fans, GPUNODE* graph, int *res, int PATTERNS,size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int val;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[graph[i].offset]];
		while (j < graph[i].nfi) {
			val = tex2D(or2LUT, val, row[fans[graph[i].offset+j]]);
			j++;
		}
		row[fans[graph[i].offset+graph[i].nfi]] = val;
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}
__global__ void NOR_gate(int i, int* fans, GPUNODE* graph, int *res, int PATTERNS,size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int val;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[graph[i].offset]];
		while (j < graph[i].nfi) {
			val = tex2D(nor2LUT, val, row[fans[graph[i].offset+j]]);
			j++;
		}
		row[fans[graph[i].offset+graph[i].nfi]] = val;
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}
__global__ void AND_gate(int i, int* fans, GPUNODE* graph, int *res, int PATTERNS,size_t width) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int val;
	while (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[graph[i].offset]];
		while (j < graph[i].nfi) {
			val = tex2D(and2LUT, val, row[fans[graph[i].offset+j]]);
			j++;
		}
		row[fans[graph[i].offset+graph[i].nfi]] = val;
		tid += blockDim.x * gridDim.x;
		__syncthreads();
	}
}
__global__ void NAND_gate(int i, int* fans, GPUNODE* graph, int *res, int PATTERNS, size_t width , int pass, int* g) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int goffset = graph[i].offset, nfi = graph[i].nfi;
	int *row;
	int val;
	if (tid < PATTERNS) {
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[goffset]];
		g[tid] = fans[goffset];
		while (j < nfi) {
			val = tex2D(nand2LUT, val, row[fans[goffset+j]]);
			j++;
		}
		if (pass > 1) {
			row[fans[goffset+nfi]] = tex2D(stableLUT, row[fans[goffset+nfi]], val);  
		} else {
			row[fans[goffset+nfi]] = val;
		}
	}
	__syncthreads();
}

__global__ void FROM_gate(int i, int* fans,GPUNODE* graph, int *res, int PATTERNS, size_t width, int pass, int* g) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, val;
	int *row;
	if (tid < PATTERNS) {
		g[tid] = fans[graph[i].offset+graph[i].nfi];
		row = (int*)((char*)res + tid*width*sizeof(int)); // get the current row?
		val = row[fans[graph[i].offset]];
		if (pass > 1) {
//			row[fans[graph[i].offset+graph[i].nfi]] = tex2D(stableLUT, row[fans[graph[i].offset+graph[i].nfi]], val);
			row[fans[graph[i].offset+graph[i].nfi]] = val;
		} else {
			row[fans[graph[i].offset+graph[i].nfi]] = val;
		}
		__syncthreads();
	}
}
__global__ void INPT_gate(int i, int pi, ARRAY2D<int> results, ARRAY2D<int> input, GPUNODE* graph, int* fans,int pass, int* g) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, val;
	int *row;
	if (tid < results.height) {
		row = (int*)((char*)results.data + tid*results.width*sizeof(int)); // get the current row?
		val = *(input.data+(pi+input.width*tid));
		g[tid] = fans[graph[i].offset+graph[i].nfi];
		if (pass > 1) {
			row[fans[graph[i].offset+graph[i].nfi]] = tex2D(stableLUT, row[fans[graph[i].offset+graph[i].nfi]], val);  
		} else {
			row[fans[graph[i].offset+graph[i].nfi]] = val;
		}
	}
#ifdef GDEBUG // turn on GPU debugging printf statements.
	printf("Hello thread %d, i=%d, input count: %d/%d input value=%d\n", threadIdx.x, i,pi+1,input.width, input.data[pi]) ;
#endif
	__syncthreads();
}

void loadLookupTables() {
	// Creating a set of static arrays that represent our LUTs
	int nand2[16] = {1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0};
	int and2[16]  = {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1};
	int nor2[16]  = {1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0};
	int or2[16]   = {0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
	int xnor2[16] = {1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1};
	int xor2[16]  = {0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0};
	int stable[4] = {S0, T0, T1, S1};
	// device memory arrays, required. 
	cudaArray *cuNandArray, *cuAndArray,*cuNorArray, *cuOrArray,*cuXnorArray,*cuXorArray, *cuStableArray;
	// generic formatting information. All of our arrays are the same, so sharing it shouldn't be a problem.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(int)*8,0,0,0,cudaChannelFormatKindUnsigned);
	
	// Allocating memory on the device.
	cudaMallocArray(&cuNandArray, &channelDesc, 4,4);
	cudaMallocArray(&cuAndArray, &channelDesc, 4,4);
	cudaMallocArray(&cuNorArray, &channelDesc, 4,4);
	cudaMallocArray(&cuOrArray, &channelDesc, 4,4);
	cudaMallocArray(&cuXnorArray, &channelDesc, 4,4);
	cudaMallocArray(&cuXorArray, &channelDesc, 4,4);
	cudaMallocArray(&cuStableArray, &channelDesc, 2,2);

	// Copying the LUTs Host->Device
	cudaMemcpyToArray(cuNandArray, 0,0, nand2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuAndArray, 0,0, and2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuNorArray, 0,0, nor2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuOrArray, 0,0, or2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXnorArray, 0,0, xnor2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXorArray, 0,0, xor2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuStableArray, 0,0, stable, sizeof(int)*4,cudaMemcpyHostToDevice);

	// Marking them as textures. LUTs should be in texture memory and cached on
	// access.
	cudaBindTextureToArray(and2LUT,cuAndArray,channelDesc);
	cudaBindTextureToArray(nand2LUT,cuNandArray,channelDesc);
	cudaBindTextureToArray(or2LUT,cuOrArray,channelDesc);
	cudaBindTextureToArray(nor2LUT,cuNorArray,channelDesc);
	cudaBindTextureToArray(xor2LUT,cuXorArray,channelDesc);
	cudaBindTextureToArray(xnor2LUT,cuXnorArray,channelDesc);
	cudaBindTextureToArray(stableLUT,cuStableArray,channelDesc);
}
void runGpuSimulation(ARRAY2D<int> results, ARRAY2D<int> inputs, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, ARRAY2D<LINE> line, int* fan, int pass = 1) {
	// Allocate a buffer memory for printf statements
	int *g;
	int piNumber = 0, curPI = 0;
	cudaMalloc((void**)&g,results.bwidth());
	cudaMemset(g, -1, sizeof(int)*results.height);
	DPRINT("Pattern Count: %d\n", results.height );
#ifndef NDEBUG
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
#endif // NDEBUG
	for (int i = 0; i <= dgraph.width; i++) {
		DPRINT("ID: %d\tFanin: %d\tFanout: %d\tType: %d\t", i, graph[i].nfi, graph[i].nfo,graph[i].type);
		curPI = piNumber;
		switch (graph[i].type) {
			case 0:
				continue;
			case INPT:
				DPRINT("INPT Gate");
				INPT_gate<<<1,results.height>>>(i, curPI, results, inputs, dgraph.data, fan, pass, g);
				piNumber++;
				break;
			case XNOR:
				DPRINT("XNOR Gate");
				XNOR_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width);
			case XOR:
				DPRINT("XOR Gate");
				XOR_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width);
			case NOR:
				DPRINT("NOR Gate");
				NOR_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width);
			case OR:
				DPRINT("OR Gate");
				OR_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width);
			case AND:
				DPRINT("AND Gate");
				AND_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width);
			case NAND:
				DPRINT("NAND Gate");
				NAND_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width, pass,g);
				break;
			case FROM:
				DPRINT("FROM Gate");
				FROM_gate<<<1,results.height>>>(i, fan, dgraph.data, results.data, results.height, results.width, pass,g);
				break;
			default:
				DPRINT("Other Gate");
				break;
		}
		DPRINT("\n");
	cudaThreadSynchronize();
	}

#ifndef NDEBUG
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	// Routine to copy contents of our results array into host memory and print
	// it row-by-row.

	DPRINT("Post-simulation device results, pass %d:\n\n", pass);
	DPRINT("Line:   \t");
	for (int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	int *lvalues, *row;
	for (int r = 0;r < results.height; r++) {
		lvalues = (int*)malloc(results.bwidth());
		row = (int*)((char*)results.data + r*results.bwidth()); // get the current row?
		cudaMemcpy(lvalues,row,results.bwidth(),cudaMemcpyDeviceToHost);
		
		DPRINT("Pattern %d:\t",r);
		for (int i = 0; i < results.width; i++) {
			switch(lvalues[i]) {
				case S0:
					DPRINT("S0 ", lvalues[i]); break;
				case S1:
					DPRINT("S1 ", lvalues[i]); break;
				case T0:
					DPRINT("T0 ", lvalues[i]); break;
				case T1:
					DPRINT("T1 ", lvalues[i]); break;
			}
		}
		DPRINT("\n");
		free(lvalues);
	}
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	DPRINT("Simulation time (pass %d): %fms\n", pass, elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif 
}

/* For each gate, color each line if a path would propagate for that TID. 
*/
void gpuColorLines(ARRAY2D<int> results, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph) {
}
