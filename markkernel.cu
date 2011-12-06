#include <cuda.h>
#include <cassert>
#include "iscas.h"
#include "defines.h"
#include "markkernel.h"

texture<int, 3> and2OutputPropLUT;
texture<int, 3> and2InputPropLUT;
texture<int, 3> or2OutputPropLUT;
texture<int, 3> or2InputPropLUT;
texture<int, 2> fromPropLUT;
texture<int, 2> inptPropLUT;
texture<int, 2> mergeLUT;

// group all results together, this implementation will fail if # of lines > 1024
// will need to group lines into groups of 1024 or less
__global__ void kernMerge(int* input, int* results, int width) {
	int *r,result, i;
	if (threadIdx.x < width) {
		result = 0;
		for (i = 0; i < blockIdx.x; i++) {
			r = (int*)((char*)input + sizeof(int)*i*width);
			result = tex2D(mergeLUT,result,r[threadIdx.x]);
		}
		r = (int*)((char*)results + sizeof(int)*width*blockIdx.x);
		r[threadIdx.x] = result;
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
	int and2_output_prop[32] ={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,0,1,1,1};
	int and2_input_prop[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1};
	int or2_output_prop[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1};
	int or2_input_prop[32] =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	int from_prop[16]      =  {0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1};
	int inpt_prop[8] = {0,0,0,0,0,0,1,1};

	cudaExtent volumeSize = make_cudaExtent(4,4,2);
	// device memory arrays, required. 
	cudaArray *cuAndInptProp, *cuAndOutpProp, *cuOrInptProp, *cuOrOutpProp, *cuFromProp, *cuInptProp;
	// generic formatting information. All of our arrays are the same, so sharing it shouldn't be a problem.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
	// Allocating memory on the device.
		
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
	cudaMemcpyToArray(cuFromProp, 0,0, from_prop, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuInptProp, 0,0, inpt_prop, sizeof(int)*8,cudaMemcpyHostToDevice);

	// Marking them as textures. LUTs should be in texture memory and cached on
	// access.
	cudaBindTextureToArray(and2OutputPropLUT,cuAndOutpProp,channelDesc);
	cudaBindTextureToArray(and2InputPropLUT,cuAndInptProp,channelDesc);
	cudaBindTextureToArray(or2OutputPropLUT,cuOrOutpProp,channelDesc);
	cudaBindTextureToArray(or2InputPropLUT,cuOrInptProp,channelDesc);
	cudaBindTextureToArray(fromPropLUT,cuFromProp,channelDesc);
	cudaBindTextureToArray(inptPropLUT,cuInptProp,channelDesc);
}

__device__ int willPathPropagate(int tid, int* results, GPUNODE* node, int* fans, size_t width) {
	return -1;
}
__global__ void kernMarkPathSegments(int *results, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
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
				
				case FROM:
					// For FROM, only set the "input" line if it hasn't already
					// been set (otherwise it'll overwrite the decision of
					// another system somewhere else.
					if (rowResults[fans[goffset]] == UNINITIALIZED) {
						val = tex2D(inptPropLUT, row[fans[goffset]],rowResults[fans[goffset+nfi]]);
						rowResults[fans[goffset]] = val;
						rowResults[fans[goffset+nfi]] = val;
					} else {
						val = tex2D(inptPropLUT, row[fans[goffset]],rowResults[fans[goffset+nfi]]);
						rowResults[fans[goffset+nfi]] = val;
					}
					break;
					// For the standard gates, setting three values -- both the input lines and the output line.
				case NAND:
				case AND:
					rowResults[fans[goffset]] = tex3D(and2InputPropLUT, row[fans[goffset]],row[fans[goffset+1]],row[fans[goffset+nfi]]-1);
					rowResults[fans[goffset+1]] = tex3D(and2InputPropLUT, row[fans[goffset+1]],row[fans[goffset]],row[fans[goffset+nfi]]-1);
					rowResults[fans[goffset+nfi]] = tex3D(and2OutputPropLUT, row[fans[goffset]],row[fans[goffset+1]],row[fans[goffset+nfi]]-1);
					break;
				case OR:
				case NOR:
					rowResults[fans[goffset]] = tex3D(or2InputPropLUT, row[fans[goffset]],row[fans[goffset+1]],row[fans[goffset+nfi]]-1);
					rowResults[fans[goffset+1]] = tex3D(or2InputPropLUT, row[fans[goffset+1]],row[fans[goffset]],row[fans[goffset+nfi]]-1);
					rowResults[fans[goffset+nfi]] = tex3D(or2OutputPropLUT, row[fans[goffset]],row[fans[goffset+1]],row[fans[goffset+nfi]]-1);
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
			row[i] = rowResults[i];// * (tid+1);
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
	kernMarkPathSegments<<<1,results.height>>>(results.data, dgraph.data, fan, results.width, results.height, dgraph.width);
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
	loadMergeLUTs();
	cudaMalloc(mergeresult, sizeof(int)*input.height*input.width);
	// for bigger circuits or more patterns, need some logic here to divide work according to what will fit. 
#ifndef NTIMING
	float elapsed;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
#endif // NTIMING
	kernMerge<<<input.height,input.width>>>(input.data, *mergeresult, input.width);
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
void runGpuSimulation(ARRAY2D<int> results, ARRAY2D<int> inputs, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, ARRAY2D<LINE> line, int* fan, int pass = 1) {
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
			DPRINT("%2c ", lvalues[i] == 0 ? 'N':'S'  );
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
