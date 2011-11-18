#ifndef GPUISCAS_H
#define GPUISCAS_H
#include <cuda.h>
#include "iscas.h"
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

int* gpuLoadVectors(int** input, size_t width, size_t height);

int* gpuLoadFans(int* offset, int maxid);
GPUNODE* gpuLoadCircuit(const GPUNODE* graph, int maxid);
LINE* gpuLoadLines(LINE* graph, int maxid);

#endif
