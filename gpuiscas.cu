#include <cuda.h>
#include "iscas.h"
#include "gpuiscas.h"
#define N 32
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

GPUNODE* gpuLoadCircuit(const GPUNODE* graph, int maxid) {
	GPUNODE *devAr;
	HANDLE_ERROR(cudaMalloc(&devAr, sizeof(GPUNODE)*maxid));
	HANDLE_ERROR(cudaMemcpy(devAr, graph, maxid * sizeof(GPUNODE),cudaMemcpyHostToDevice));
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

int* gpuLoadVectors(int** input, size_t width, size_t height) {
	int *tgt;
	HANDLE_ERROR(cudaMalloc(&tgt, sizeof(int)*width*height));
	int *row;
	int *tmp = (int*)malloc(sizeof(int)*width);
	for (int i =0; i < width; i++)
		tmp[i] = -1;
	for (int i = 0; i < height; i++) {
		row = (int*)((char*)tgt + i*width*sizeof(int));
		cudaMemcpy(row, input[i],sizeof(int)*width,cudaMemcpyHostToDevice);
//		cudaMemcpy(tmp, row, sizeof(int)*width,cudaMemcpyDeviceToHost);
//		printf("Checking copy results:\n");
//		for (int j = 0; j < width; j++) {
//			printf("(%d,%d) ",input[i][j], tmp[j]);
//		}
//		printf("\n");
	}
	return tgt;
}
