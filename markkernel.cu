#include <cuda.h>
#include <cassert>
#include "iscas.h"
#include "defines.h"
#include "markkernel.h"

#define THREAD_PER_BLOCK 256
texture<int, 3> and2OutputPropLUT;
texture<int, 3> and2InputPropLUT;
texture<int, 3> or2OutputPropLUT;
texture<int, 3> or2InputPropLUT;
texture<int, 3> xor2OutputPropLUT;
texture<int, 3> xor2InputPropLUT;
texture<int, 2> fromPropLUT;
texture<int, 2> inptPropLUT;
texture<int, 2> mergeLUT;

__device__ void faninRemake(int *results, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {

}
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
	int xor2_input_prop[32] =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	int xor2_output_prop[32] =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	int from_prop[16]      =  {0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1};
	int inpt_prop[8] = {0,0,0,0,0,0,1,1};

	cudaExtent volumeSize = make_cudaExtent(4,4,2);
	// device memory arrays, required. 
	cudaArray *cuAndInptProp, *cuAndOutpProp, *cuOrInptProp, *cuOrOutpProp, *cuFromProp, *cuInptProp, *cuXorInptProp, *cuXorOutpProp;
	// generic formatting information. All of our arrays are the same, so sharing it shouldn't be a problem.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
	// Allocating memory on the device.
		
	cudaMallocArray(&cuFromProp, &channelDesc, 4,4);
	cudaMallocArray(&cuInptProp, &channelDesc, 4,2);
	cudaMalloc3DArray(&cuAndInptProp, &channelDesc, volumeSize);
	cudaMalloc3DArray(&cuAndOutpProp, &channelDesc, volumeSize);
	cudaMalloc3DArray(&cuOrInptProp, &channelDesc, volumeSize);
	cudaMalloc3DArray(&cuOrOutpProp, &channelDesc, volumeSize);
	cudaMalloc3DArray(&cuXorInptProp, &channelDesc, volumeSize);
	cudaMalloc3DArray(&cuXorOutpProp, &channelDesc, volumeSize);

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

	copyParams.srcPtr = make_cudaPitchedPtr((void*)xor2_input_prop, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
	copyParams.dstArray = cuXorInptProp;
	copyParams.extent =  volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);
	or2OutputPropLUT.addressMode[2]=cudaAddressModeClamp;
	or2OutputPropLUT.addressMode[0]=cudaAddressModeClamp;
	or2OutputPropLUT.addressMode[1]=cudaAddressModeClamp;
	// Copying the LUTs Host->Device
	cudaMemcpyToArray(cuFromProp, 0,0, from_prop, sizeof(int)*16,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuInptProp, 0,0, inpt_prop, sizeof(int)*8,cudaMemcpyHostToDevice);

	copyParams.srcPtr = make_cudaPitchedPtr((void*)xor2_output_prop, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
	copyParams.dstArray = cuXorOutpProp;
	copyParams.extent =  volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);
	or2OutputPropLUT.addressMode[2]=cudaAddressModeClamp;
	or2OutputPropLUT.addressMode[0]=cudaAddressModeClamp;
	or2OutputPropLUT.addressMode[1]=cudaAddressModeClamp;
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
}

__device__ int willPathPropagate(int tid, int* results, GPUNODE* node, int* fans, size_t width) {
	return -1;
}
__global__ void kernMarkPathSegments(int *results, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x, nfi, goffset,val, type;
	__shared__ char rowids[1000]; // handle up to fanins of 1000 / 
	int *rowResults, *row;
	__shared__ int fan[20];
	if (tid < height) {
		row = (int*)((char*)results + tid*(width)*sizeof(int));
		rowResults = (int*)malloc(sizeof(int)*width);
		for (int i = 0; i < width; i++) {
			rowResults[i] = UNINITIALIZED;
		}
		for (int i = ncount; i >= 0; i--) {
			if (tid == 0) {
				goffset = node[i].offset;
				// preload all of the fanin line #s for this gate to shared memory.
				for (int j = 0; j < nfi;j++) 
					rowids[j] = (char)fans[goffset+j];
			}
			__syncthreads();
			val = UNINITIALIZED;
			goffset = node[i].offset;
			for (int j = 0; j < nfi; j++) {
				fan[j] = fans[goffset+j];
			}
			// switching based on value causes divergence, switch based on node type.
			switch(node[i].type) {
				case FROM:
					// For FROM, only set the "input" line if it hasn't already
					// been set (otherwise it'll overwrite the decision of
					// another system somewhere else.
					if (rowResults[fans[goffset]] == UNINITIALIZED) {
						val = tex2D(inptPropLUT, row[rowids[0]],rowResults[i]);
						rowResults[rowids[0]] = val;
						rowResults[i] = val;
					} else {
						val = tex2D(inptPropLUT, row[fans[goffset]],rowResults[i]);
						rowResults[i] = val;
					}
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
					rowResults[rowids[0]] = tex3D(and2InputPropLUT, row[rowids[0]],row[rowids[1]],row[i]-1);
					rowResults[rowids[1]] = tex3D(and2InputPropLUT, row[rowids[1]],row[rowids[0]],row[i]-1);
					rowResults[i] = tex3D(and2OutputPropLUT, row[rowids[0]],row[rowids[1]],row[i]-1);
					break;
				case OR:
				case NOR:
					rowResults[rowids[0]] = tex3D(or2InputPropLUT, row[rowids[0]],row[rowids[1]],row[i]-1);
					rowResults[rowids[1]] = tex3D(or2InputPropLUT, row[rowids[1]],row[rowids[0]],row[i]-1);
					rowResults[i] = tex3D(or2OutputPropLUT, row[rowids[0]],row[rowids[1]],row[i]-1);
					break;
				case XOR:
				case XNOR:
					rowResults[rowids[0]] = tex3D(xor2InputPropLUT, row[rowids[0]],row[rowids[1]],row[i]-1);
					rowResults[rowids[1]] = tex3D(xor2InputPropLUT, row[rowids[1]],row[rowids[0]],row[i]-1);
					rowResults[i] = tex3D(xor2OutputPropLUT, row[rowids[0]],row[rowids[1]],row[i]-1);
				case BUFF:
				case NOT:
						val = tex2D(inptPropLUT, row[rowids[0]],rowResults[i]);
						rowResults[rowids[0]] = val;
						rowResults[i] = val;
					break;
				default:
					// if there is a transition that will propagate, set = to some positive #?
					break;
			}
		}
		// replace our working set.
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
#endif
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
