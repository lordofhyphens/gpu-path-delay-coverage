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

NODE* gpuLoadCircuit(const NODE* graph, int maxid) {
	NODE *devAr;
	HANDLE_ERROR(cudaMalloc(&devAr, sizeof(NODE)*maxid));
	HANDLE_ERROR(cudaMemcpy(devAr, graph, maxid * sizeof(NODE),cudaMemcpyHostToDevice));
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
	size_t pitch;
	HANDLE_ERROR(cudaMallocPitch(&tgt, &pitch, width, height));
	HANDLE_ERROR(cudaMemcpy2D(tgt,pitch,input,pitch,width,height,cudaMemcpyHostToDevice));
	return tgt;
}
