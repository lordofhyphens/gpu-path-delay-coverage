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
texture<int, 3> and2OutputPropLUT;
texture<int, 3> and2InputPropLUT;
texture<int, 3> or2OutputPropLUT;
texture<int, 3> or2InputPropLUT;
texture<int, 2> fromPropLUT;
texture<int, 2> inptPropLUT;

__global__ void INPT_gate(int i, int pi, ARRAY2D<int> results, ARRAY2D<int> input, GPUNODE* graph, int* fans,int pass) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, val;
	int *row;
	if (tid < results.height) {
		row = (int*)((char*)results.data + tid*results.width*sizeof(int)); // get the current row?
		val = *(input.data+(pi+input.width*tid));
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
	// Addressing for the propagations:
	// 2 4x4 groups such that 
	int and2_output_prop[32] ={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,0,0,0, 1, 1, 1, 1, 1, 1,0, 1, 1, 1};
	int and2_input_prop[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1, 1,0,0, 1,0,0,0, 1, 1};
	int or2_output_prop[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1, 1,0,0,0,0, 1,0, 1, 1, 1,0, 1, 1};
	int or2_input_prop[32] =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1, 1,0,0,0,0,0,0, 1, 1,0,0,0, 1};
	int from_prop[16]      =  {0,0,0,0,0,0,0,0,0,0, 1, 1,0,0, 1, 1};
	int inpt_prop[8] = {0,0,0,0,0,0,1,1};

	cudaExtent volumeSize = make_cudaExtent(4,4,2);
	// device memory arrays, required. 
	cudaArray *cuNandArray, *cuAndArray,*cuNorArray, *cuOrArray,*cuXnorArray,*cuXorArray, *cuStableArray;
	cudaArray *cuAndInptProp, *cuAndOutpProp, *cuOrInptProp, *cuOrOutpProp, *cuFromProp, *cuInptProp;
	// generic formatting information. All of our arrays are the same, so sharing it shouldn't be a problem.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
	DPRINT("%d,%d", volumeSize.height, volumeSize.width);
	
	// Allocating memory on the device.
	cudaMallocArray(&cuNandArray, &channelDesc, 4,4);
	cudaMallocArray(&cuAndArray, &channelDesc, 4,4);
	cudaMallocArray(&cuNorArray, &channelDesc, 4,4);
	cudaMallocArray(&cuOrArray, &channelDesc, 4,4);
	cudaMallocArray(&cuXnorArray, &channelDesc, 4,4);
	cudaMallocArray(&cuXorArray, &channelDesc, 4,4);
	cudaMallocArray(&cuStableArray, &channelDesc, 2,2);
	
	cudaMallocArray(&cuFromProp, &channelDesc, 4,4);
	cudaMallocArray(&cuInptProp, &channelDesc, 4,2);
	cudaMalloc3DArray(&cuAndInptProp, &channelDesc, volumeSize);
	cudaMalloc3DArray(&cuAndOutpProp, &channelDesc, volumeSize);
	cudaMalloc3DArray(&cuOrInptProp, &channelDesc, volumeSize);
	cudaMalloc3DArray(&cuOrOutpProp, &channelDesc, volumeSize);

	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr((void*)and2_output_prop, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
	copyParams.dstArray = cuAndOutpProp;
	copyParams.extent =  volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);
	and2OutputPropLUT.addressMode[2]=cudaAddressModeClamp;
	and2OutputPropLUT.addressMode[0]=cudaAddressModeClamp;
	and2OutputPropLUT.addressMode[1]=cudaAddressModeClamp;

	copyParams.srcPtr = make_cudaPitchedPtr((void*)and2_input_prop, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
	copyParams.dstArray = cuAndInptProp;
	copyParams.extent =  volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);
	and2OutputPropLUT.addressMode[2]=cudaAddressModeClamp;
	and2OutputPropLUT.addressMode[0]=cudaAddressModeClamp;
	and2OutputPropLUT.addressMode[1]=cudaAddressModeClamp;

	copyParams.srcPtr = make_cudaPitchedPtr((void*)or2_output_prop, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
	copyParams.dstArray = cuOrOutpProp;
	copyParams.extent =  volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);
	or2OutputPropLUT.addressMode[2]=cudaAddressModeClamp;
	or2OutputPropLUT.addressMode[0]=cudaAddressModeClamp;
	or2OutputPropLUT.addressMode[1]=cudaAddressModeClamp;

	copyParams.srcPtr = make_cudaPitchedPtr((void*)or2_input_prop, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
	copyParams.dstArray = cuOrInptProp;
	copyParams.extent =  volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);
	or2OutputPropLUT.addressMode[2]=cudaAddressModeClamp;
	or2OutputPropLUT.addressMode[0]=cudaAddressModeClamp;
	or2OutputPropLUT.addressMode[1]=cudaAddressModeClamp;

	// Copying the LUTs Host->Device
	cudaMemcpyToArray(cuNandArray, 0,0, nand2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuAndArray, 0,0, and2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuNorArray, 0,0, nor2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuOrArray, 0,0, or2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXnorArray, 0,0, xnor2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuXorArray, 0,0, xor2, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuFromProp, 0,0, from_prop, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuInptProp, 0,0, inpt_prop, sizeof(int)*8,cudaMemcpyHostToDevice);
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
	cudaBindTextureToArray(and2OutputPropLUT,cuAndOutpProp,channelDesc);
	cudaBindTextureToArray(and2InputPropLUT,cuAndInptProp,channelDesc);
	cudaBindTextureToArray(or2OutputPropLUT,cuOrOutpProp,channelDesc);
	cudaBindTextureToArray(or2InputPropLUT,cuOrInptProp,channelDesc);
	cudaBindTextureToArray(fromPropLUT,cuFromProp,channelDesc);
	cudaBindTextureToArray(inptPropLUT,cuInptProp,channelDesc);
}

__global__ void LOGIC_gate(int i, GPUNODE* node, int* fans, int* res, size_t height, size_t width , int pass) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, j = 1;
	int *row;
	int goffset,nfi;
	int val;
	if (tid < height) {
		goffset = node[i].offset;
		nfi = node[i].nfi;
		row = (int*)((char*)res + tid*(width)*sizeof(int));
		val = row[fans[goffset]];
		while (j < nfi) {
			switch(node[i].type) {
				case XOR:
					val = tex2D(xor2LUT, val, row[fans[goffset+j]]);
				case XNOR:
					val = tex2D(xnor2LUT, val, row[fans[goffset+j]]);
				case OR:
					val = tex2D(or2LUT, val, row[fans[goffset+j]]);
				case NOR:
					val = tex2D(nor2LUT, val, row[fans[goffset+j]]);
				case AND:
					val = tex2D(and2LUT, val, row[fans[goffset+j]]);
				case NAND:
					val = tex2D(nand2LUT, val, row[fans[goffset+j]]);
			}
			j++;
		}
		if (pass > 1 && node[i].type != FROM) {
			row[fans[goffset+nfi]] = tex2D(stableLUT, row[fans[goffset+nfi]], val);  
		} else {
			row[fans[goffset+nfi]] = val;
		}
	}
	__syncthreads();
}
__device__ int willPathPropagate(int tid, int* results, GPUNODE* node, int* fans, size_t width) {
	return -1;
}
__global__ void gpuMarkPathSegments(int *results, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x, nfi, goffset,val;
	int *rowResults, *row;
	if (tid < height) {
		row = (int*)((char*)results + tid*(width)*sizeof(int));
		rowResults = (int*)malloc(sizeof(int)*width);
		for (int i = 0; i < width; i++) {
			rowResults[i] = UNINITIALIZED;
		}
		for (int i = ncount; i >= 0; i--) {
			val = UNINITIALIZED;
			goffset = node[i].offset;
			nfi = node[i].nfi;
			// switching based on value causes divergence, switch based on node type.
			switch(node[i].type) {
				case NAND:
					rowResults[fans[goffset]] = tex3D(and2InputPropLUT, row[fans[goffset]],row[fans[goffset+1]],row[fans[goffset+nfi]]-1);
					rowResults[fans[goffset+1]] = tex3D(and2InputPropLUT, row[fans[goffset+1]],row[fans[goffset]],row[fans[goffset+nfi]]-1) ;
					rowResults[fans[goffset+nfi]] = tex3D(and2OutputPropLUT, row[fans[goffset]],row[fans[goffset+1]],row[fans[goffset+nfi]]-1) ;
					break;
				case FROM:
					if (rowResults[fans[goffset]] == UNINITIALIZED) {
						val = tex2D(inptPropLUT, row[fans[goffset]],rowResults[fans[goffset+nfi]]);
						rowResults[fans[goffset]] = val;
						rowResults[fans[goffset+nfi]] = val;
					} else {
						val = tex2D(inptPropLUT, row[fans[goffset]],rowResults[fans[goffset+nfi]]);
						rowResults[fans[goffset+nfi]] = val;
					}
					break;
				case AND:
					rowResults[fans[goffset]] = tex3D(and2InputPropLUT, row[fans[goffset]],row[fans[goffset+1]],row[fans[goffset+nfi]]-1);
					rowResults[fans[goffset+1]] = tex3D(and2InputPropLUT, row[fans[goffset+1]],row[fans[goffset]],row[fans[goffset+nfi]]-1);
					rowResults[fans[goffset+nfi]] = tex3D(and2OutputPropLUT, row[fans[goffset]],row[fans[goffset+1]],row[fans[goffset+nfi]]-1);
					break;
				case OR:
					rowResults[fans[goffset]] = tex3D(or2InputPropLUT, row[fans[goffset]],row[fans[goffset+1]],row[fans[goffset+nfi]]-1) ;
					rowResults[fans[goffset+1]] = tex3D(or2InputPropLUT, row[fans[goffset+1]],row[fans[goffset]],row[fans[goffset+nfi]]-1) ;
					rowResults[fans[goffset+nfi]] = tex3D(or2OutputPropLUT, row[fans[goffset]],row[fans[goffset+1]],row[fans[goffset+nfi]]-1) ;
					break;
				case NOR:
					rowResults[fans[goffset]] = tex3D(or2InputPropLUT, row[fans[goffset]],row[fans[goffset+1]],row[fans[goffset+nfi]]-1) ;
					rowResults[fans[goffset+1]] = tex3D(or2InputPropLUT, row[fans[goffset+1]],row[fans[goffset]],row[fans[goffset+nfi]]-1) ;
					rowResults[fans[goffset+nfi]] = tex3D(or2OutputPropLUT, row[fans[goffset]],row[fans[goffset+1]],row[fans[goffset+nfi]]-1) ;
					break;
				case XOR:
				case XNOR:
				default:
					// if there is a transition that will propagate, set = to some positive #?
					break;
			}
		}
		__syncthreads();
		for (int i = 0; i < width; i++) {
			row[i] = rowResults[i] * (tid+1);
		}
		free(rowResults);
	}
}
void runGpuSimulation(ARRAY2D<int> results, ARRAY2D<int> inputs, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, ARRAY2D<LINE> line, int* fan, int pass = 1) {
	// Allocate a buffer memory for printf statements
	int piNumber = 0, curPI = 0;
	DPRINT("Pattern Count: %d\n", results.height );
#ifndef NDEBUG
	int eventCount = 0;
	cudaEvent_t* eventQueue = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*10);
	cudaEventCreate(eventQueue);
	cudaEventCreate(eventQueue+1);
	eventCount++;
	cudaEventRecord(eventQueue[0],0);
#endif // NDEBUG
	for (int i = 0; i <= dgraph.width; i++) {
		DPRINT("ID: %d\tFanin: %d\tFanout: %d\tType: %d\t", i, graph[i].nfi, graph[i].nfo,graph[i].type);
		curPI = piNumber;
		switch (graph[i].type) {
			case 0:
				continue;
			case INPT:
				DPRINT("INPT Gate");
				INPT_gate<<<1,results.height>>>(i, curPI, results, inputs, dgraph.data, fan, pass);
				piNumber++;
				break;
			default:
				LOGIC_gate<<<1,results.height>>>(i, dgraph.data, fan, results.data, results.height, results.width, pass);
				break;
		}
		DPRINT("\n");
		cudaDeviceSynchronize();
	}
#ifndef NDEBUG
	cudaEventRecord(eventQueue[1], 0);
	cudaEventSynchronize(eventQueue[1]);

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
		
		DPRINT("%s %d:\t", pass > 1 ? "Vector" : "Pattern",r);
		for (int i = 0; i < results.width; i++) {
				switch(lvalues[i]) {
					case S0:
						DPRINT("S0 "); break;
					case S1:
						DPRINT("S1 "); break;
					case T0:
						DPRINT("T0 "); break;
					case T1:
						DPRINT("T1 "); break;
					default:
						DPRINT("%2d ", lvalues[i]); break;
				}

		}
		DPRINT("\n");
		free(lvalues);
	}

	cudaEventCreate(eventQueue+2);
	cudaEventCreate(eventQueue+3);
	eventCount++;
	cudaEventRecord(eventQueue[2],0);
#endif
	if (pass > 1) {
		gpuMarkPathSegments<<<1,results.height>>>(results.data, dgraph.data, fan, results.width, results.height, dgraph.width);
		cudaDeviceSynchronize();
	}
#ifndef NDEBUG
	cudaEventRecord(eventQueue[3],0);
	cudaEventSynchronize(eventQueue[3]);
	// Routine to copy contents of our results array into host memory and print
	// it row-by-row.

	DPRINT("Post-simulation device results, pass %d:\n\n", pass);
	DPRINT("Line:   \t");
	for (int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (int r = 0;r < results.height; r++) {
		lvalues = (int*)malloc(results.bwidth());
		row = (int*)((char*)results.data + r*results.bwidth()); // get the current row?
		cudaMemcpy(lvalues,row,results.bwidth(),cudaMemcpyDeviceToHost);
		
		DPRINT("%s %d:\t", pass > 1 ? "Vector" : "Pattern",r);
		for (int i = 0; i < results.width; i++) {
			if (pass > 1) {
				DPRINT("%2c ", lvalues[i] == 0 ? 'N':'S'  );
			} else {
				switch(lvalues[i]) {
					case S0:
						DPRINT("S0 "); break;
					case S1:
						DPRINT("S1 "); break;
					case T0:
						DPRINT("T0 "); break;
					case T1:
						DPRINT("T1 "); break;
					default:
						DPRINT("%2d ", lvalues[i]); break;
				}
			}

		}
		DPRINT("\n");
		free(lvalues);
	}
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,eventQueue[0],eventQueue[1]);
	DPRINT("Simulation time (pass %d): %fms\n", pass, elapsedTime);
	if (pass > 1) {
		cudaEventElapsedTime(&elapsedTime,eventQueue[2],eventQueue[3]);
		DPRINT("Path Marking time (pass %d): %fms\n", pass, elapsedTime);
	}
	for (int f = 0; f < eventCount*2; f+=2) {
		cudaEventDestroy(eventQueue[f]);
		cudaEventDestroy(eventQueue[f+1]);
	}
	free(eventQueue);
#endif 
}

