#include "gpuiscas.h"

void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
		if (err == cudaErrorInvalidValue)
			DPRINT("cudaErrorInvalidValue: ");
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            DPRINT("Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

GPUNODE* gpuLoadCircuit(const GPUNODE* graph, int maxid) {
	GPUNODE *devAr, *testAr;
	HANDLE_ERROR(cudaMalloc(&devAr, sizeof(GPUNODE)*(maxid)));
	HANDLE_ERROR(cudaMemcpy(devAr, graph, (maxid) * sizeof(GPUNODE),cudaMemcpyHostToDevice));
//	DPRINT("Verifying GPUNODE graph copy\n");
	testAr = (GPUNODE*)malloc(sizeof(GPUNODE)*(maxid));	
	HANDLE_ERROR(cudaMemcpy(testAr, devAr, (maxid) * sizeof(GPUNODE),cudaMemcpyDeviceToHost));

	for (int i = 0; i < maxid; i++) {
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
	int* devAr = NULL;
	cudaError_t returncode;
	returncode = cudaMalloc(&devAr, sizeof(int)*maxid);
	assert(returncode == cudaSuccess);
	assert(devAr != NULL);
	returncode = cudaMemcpy(devAr, offset, sizeof(int)*maxid,cudaMemcpyHostToDevice);
	assert(returncode == cudaSuccess);
#ifndef NDEBUG
		int *tmp = (int*)malloc(sizeof(int)*maxid);
		assert(tmp != NULL);
		for (int r =0; r < maxid;r++)
			tmp[r] = -1;
		cudaMemcpy(tmp, devAr, sizeof(int)*(maxid),cudaMemcpyDeviceToHost);
		for (int i = 0; i < maxid; i++) {
			assert(offset[i]==tmp[i]);
		}
		free(tmp);
#endif // debugging memory check and assertion
	return devAr;
}
void gpuShiftVectors(int* input, size_t width, size_t height) {
	int* tgt = NULL;
	// create a temporary buffer area on the device
	HANDLE_ERROR(cudaMalloc(&tgt, sizeof(int)*(width)));
	HANDLE_ERROR(cudaMemcpy(tgt, input,sizeof(int)*(width),cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(input, input+width,sizeof(int)*(width)*(height-1),cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(input+(height-1)*(width),tgt, sizeof(int)*(width), cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaFree(tgt));
}
ARRAY2D<int> gpuAllocateResults(size_t width, size_t height) {
	int *tgt = NULL;
	size_t pitch = 0;
	DPRINT("Attempting to allocate %u * %u = %lu bytes... %G megabytes ",(int)sizeof(unsigned)*(unsigned)width,(unsigned)height, sizeof(int)*width*height, sizeof(int)*width*height / pow(2,20));
	HANDLE_ERROR(cudaMallocPitch(&tgt, &pitch, sizeof(int)*(width),height));
	DPRINT("...complete.\n");
	DPRINT("Allocated %u*%u = %lu bytes, %G megabytes\n", (unsigned)pitch,(unsigned)height, pitch*height, (pitch*width)/pow(2,20));
	HANDLE_ERROR(cudaMemset2D(tgt, pitch,0, sizeof(int)*width,height));
	return ARRAY2D<int>(tgt, height, width, pitch);
}
int* gpuLoad1DVector(int* input, size_t width, size_t height) {
	int *tgt;
	HANDLE_ERROR(cudaMalloc(&tgt, sizeof(int)*(width)*(height)));
	HANDLE_ERROR(cudaMemcpy(tgt, input,sizeof(int)*(width)*(height),cudaMemcpyHostToDevice));
	return tgt;
}
int* loadPinned(int* input, size_t vcnt) {
	int* tgt;
	cudaMallocHost(&tgt, vcnt*sizeof(int));
	cudaMemcpy(tgt, input, sizeof(int)*vcnt, cudaMemcpyHostToHost);
	return tgt;
}
void freeMemory(int* data) {
	cudaFree(data);
}
void freeMemory(GPUNODE* data) {
	cudaFree(data);
}

