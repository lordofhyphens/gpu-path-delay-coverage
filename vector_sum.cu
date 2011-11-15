#include <cuda.h>
#include <stdio.h>
#define N (33*1024)
#define true 1
#define false 0
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
__global__ void add(int *a, int *b, int *c) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}
int main(void) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	HANDLE_ERROR ( cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR ( cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR ( cudaMalloc((void**)&dev_c, N * sizeof(int)));

	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i*i;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a,a,N*sizeof(int),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b,b,N*sizeof(int),cudaMemcpyHostToDevice));

	add<<<128,128>>>(dev_a,dev_b,dev_c);

	HANDLE_ERROR(cudaMemcpy(c,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost));
	int success = true;
	for (int i = 0; i < N; i++) {
		if ((a[i]+b[i])!=c[i]){
			printf("Error: %d + %d != %d\n",a[i],b[i],c[i]);
			success=false;
		}
	}
	if (success)
		printf("All numbers added correctly.\n");
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
