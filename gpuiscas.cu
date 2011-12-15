#include <cuda.h>
#include "iscas.h"
#include "gpuiscas.h"
#define NDEBUG
#include "defines.h"
#include <cassert>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf("Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

GPUNODE* gpuLoadCircuit(const GPUNODE* graph, int maxid) {
	GPUNODE *devAr, *testAr;
	HANDLE_ERROR(cudaMalloc(&devAr, sizeof(GPUNODE)*(1+maxid)));
	HANDLE_ERROR(cudaMemcpy(devAr, graph, (maxid+1) * sizeof(GPUNODE),cudaMemcpyHostToDevice));
	DPRINT("Verifying GPUNODE graph copy\n");
	testAr = (GPUNODE*)malloc(sizeof(GPUNODE)*(maxid+1));	
	HANDLE_ERROR(cudaMemcpy(testAr, devAr, (1+maxid) * sizeof(GPUNODE),cudaMemcpyDeviceToHost));

	for (int i = 0; i <= maxid; i++) {
		assert(testAr[i].type == graph[i].type && testAr[i].nfi == graph[i].nfi &&testAr[i].nfo == graph[i].nfo && testAr[i].po == graph[i].po && testAr[i].offset == graph[i].offset);
	}
	free(testAr);
	return devAr;
}
LINE* gpuLoadLines(LINE* graph, int maxid) {
	LINE *devAr;
	HANDLE_ERROR(cudaMalloc(&devAr, sizeof(LINE)*maxid));
	HANDLE_ERROR(cudaMemcpy(devAr, graph, sizeof(LINE)*maxid,cudaMemcpyHostToDevice));
	return devAr;
}
int* gpuLoadFans(int* offset, int maxid) {
	int* devAr;
	HANDLE_ERROR(cudaMalloc(&devAr, sizeof(int)*maxid));
	HANDLE_ERROR(cudaMemcpy(devAr, offset, sizeof(int)*maxid,cudaMemcpyHostToDevice));
	return devAr;
}
void gpuShiftVectors(int* input, size_t width, size_t height) {
	int* tgt;
	// create a temporary buffer area on the device
	HANDLE_ERROR(cudaMalloc(&tgt, sizeof(int)*(width)));
	HANDLE_ERROR(cudaMemcpy(tgt, input,sizeof(int)*(width),cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(input, input+width,sizeof(int)*(width)*(height-1),cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(input+(height-1)*(width),tgt, sizeof(int)*(width), cudaMemcpyDeviceToDevice));
	cudaFree(tgt);
}
int* gpuLoadVectors(int** input, size_t width, size_t height) {
	int *tgt;
	HANDLE_ERROR(cudaMalloc(&tgt, sizeof(int)*(width+1)*(height+1)));
	int *row;
	for (int i = 0; i < height; i++) {
		row = (int*)((char*)tgt + i*(width)*sizeof(int));
		cudaMemcpy(row, input[i],sizeof(int)*(width+1),cudaMemcpyHostToDevice);
#ifndef NDEBUG
		int *tmp = (int*)malloc(sizeof(int)*width);
		for (int i =0; i <= width; i++)
			tmp[i] = -1;
		cudaMemcpy(tmp, row, sizeof(int)*(width+1),cudaMemcpyDeviceToHost);
		for (int j = 0; j <= width; j++) {
			assert(input[i][j]==tmp[j]);
		}
		free(tmp);
#endif // debugging memory check and assertion
	}
	return tgt;
}
int* gpuLoad1DVector(int* input, size_t width, size_t height) {
	int *tgt;
	HANDLE_ERROR(cudaMalloc(&tgt, sizeof(int)*(width+1)*(height+1)));
	cudaMemcpy(tgt, input,sizeof(int)*(width+1)*(height+1),cudaMemcpyHostToDevice);
	return tgt;
}
