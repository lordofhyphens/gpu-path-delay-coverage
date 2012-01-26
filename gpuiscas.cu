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
void gpuShiftVectors(ARRAY2D<int> input) {
	int* tgt = NULL;
	int* tgt2 = NULL;
	char* src = (char*)input.data + input.pitch;
	char* dst = (char*)input.data + (input.height-1)*input.pitch;
	// create a temporary buffer area on the device
	HANDLE_ERROR(cudaMalloc(&tgt, input.pitch));
	HANDLE_ERROR(cudaMalloc(&tgt2, input.pitch*(input.height-1)));

	HANDLE_ERROR(cudaMemcpy(tgt,  input.data,            input.pitch,cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(tgt2, src,(input.pitch)*(input.height-1),cudaMemcpyDeviceToDevice));

	HANDLE_ERROR(cudaMemcpy(input.data, tgt2, input.pitch*(input.height-1),cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(dst,tgt, input.pitch, cudaMemcpyDeviceToDevice));

	HANDLE_ERROR(cudaFree(tgt));
	HANDLE_ERROR(cudaFree(tgt2));
}
ARRAY2D<char> gpuAllocateResults(size_t width, size_t height) {
	char *tgt = NULL;
	size_t pitch = 0;
	DPRINT("Attempting to allocate %u * %u = %lu bytes... %G megabytes ",(int)sizeof(char)*(unsigned)height,(unsigned)width, sizeof(char)*width*height, (sizeof(char)*width*height) / pow(2,20));
	HANDLE_ERROR(cudaMallocPitch(&tgt, &pitch, sizeof(char)*(height),width));
	DPRINT("...complete.\n");
	DPRINT("Allocated %u*%u = %lu bytes, %G megabytes\n", (unsigned)pitch,(unsigned)width, pitch*width, ((pitch*width) / pow(2,20)));
	HANDLE_ERROR(cudaMemset2D(tgt, pitch,0, sizeof(char)*height,width));
	HANDLE_ERROR(cudaGetLastError());
	return ARRAY2D<char>(tgt, height, width, pitch);
}
ARRAY2D<int> gpuAllocateBlockResults(size_t height) {
	int* tgt = NULL;
	HANDLE_ERROR(cudaMalloc(&tgt, sizeof(int)*(height)));
	HANDLE_ERROR(cudaMemset(tgt, -1, sizeof(int)*height));
	return ARRAY2D<int>(tgt, height, 1);
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
	HANDLE_ERROR(cudaFree(data));
}
void freeMemory(char* data) {
	HANDLE_ERROR(cudaFree(data));
}
void freeMemory(GPUNODE* data) {
	HANDLE_ERROR(cudaFree(data));
}
void clearMemory(ARRAY2D<char> ar) {
	HANDLE_ERROR(cudaMemset2D(ar.data, ar.pitch,0, sizeof(char)*ar.height,ar.width));
}
void gpuPrintVectors(int* vec, size_t height, size_t width) {
	int* tmp = (int*)malloc(sizeof(int)*width*height);
	cudaMemcpy(tmp, vec, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	for (unsigned int i = 0; i < height; i++) {
		for (unsigned int j = 0; j < width; j++) {
			printf("%d", tmp[(i*width) + j]);
		}
		printf("\n");
	}
}
void gpu1PrintVectors(int* vec, size_t height, size_t width) {
	int* tmp = (int*)malloc(sizeof(int)*width*height);
	cudaMemcpy(tmp, vec, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	for (unsigned int i = 0; i < height; i++) {
		for (unsigned int j = 0; j < width; j++) {
			printf("%d ", tmp[(i*width) + j]);
		}
		printf("\n");
	}
}
void debugPrintVectors(ARRAY2D<int> results) {
#ifndef NDEBUG
int *lvalues, *row;
DPRINT("Primary Input Vectors\n");
DPRINT("Gate:   \t");
for (unsigned int i = 0; i < results.width; i++) {
	DPRINT("%2d ", i);
}
DPRINT("\n");
for (unsigned int r = 0;r < results.height; r++) {
	lvalues = (int*)malloc(results.pitch);
	row = (int*)((char*)results.data + r*results.pitch); // get the current row?
	cudaMemcpy(lvalues,row,results.width,cudaMemcpyDeviceToHost);
	DPRINT("%s %d:\t", "Vector ",r);
	for (unsigned int i = 0; i < results.width; i++) {
		switch(lvalues[i]) {
			case S0:
				DPRINT(" 0 "); break;
			case S1:
				DPRINT(" 1 "); break;
			default:
				DPRINT("%2d ", lvalues[i]); break;
		}
	}
	DPRINT("\n");
	free(lvalues);
}
#endif
}
void gpuArrayCopy(ARRAY2D<char> dst, ARRAY2D<char> src) {
	HANDLE_ERROR(cudaMemcpy2D(dst.data, dst.pitch, src.data, src.pitch, sizeof(char)*src.height, src.width, cudaMemcpyDeviceToDevice));
}
