#include <cuda.h>
#include "gpudata.h"
#include "defines.h"

GPU_Data::GPU_Data() {
	this->_gpu = new ARRAY2D<char>();
	this->_block_size = 0;
}
GPU_Data::GPU_Data(size_t rows, size_t columns) {
	this->_gpu = new ARRAY2D<char>();
	this->initialize(rows, columns, rows);
}
GPU_Data::GPU_Data(size_t rows, size_t columns, int blockwidth) {
	this->_gpu = new ARRAY2D<char>();
	this->initialize(rows, columns, blockwidth);
}
GPU_Data::~GPU_Data() {
	if (this->_gpu->data != NULL) {
		cudaFree(this->_gpu->data);
	}
}
ARRAY2D<char> GPU_Data::gpu(int ref) {
	if (ref == this->_current) {
		return *(this->_gpu);
	}
	int tmp = this->_current;
	int err; 
	try {
		err = this->copy(ref);
	} catch (std::out_of_range& oor) { 
		// handling the problem by returning NULL and ensuring that _current is not changed.
		this->_current = tmp;
		return ARRAY2D<char>(NULL,0,0,0);
	}
	if (err != ERR_NONE)
		return ARRAY2D<char>(NULL,0,0,0);
	return *(this->_gpu);
}

// total size in columns, rows. 
int GPU_Data::initialize(size_t in_columns, size_t in_rows, int block_width) {
	int chunks = (in_columns / block_width) + ((in_columns % block_width) > 0);

	this->_gpu = new ARRAY2D<char>(NULL, in_rows, block_width, sizeof(char)*block_width);
	cudaMallocPitch(&(this->_gpu->data), &(this->_gpu->pitch), sizeof(char)*this->_gpu->width, in_rows);
	int rem_columns = in_columns;
	for (int i = 0; i < chunks;i++) {
		this->_data->push_back(ARRAY2D<char>(in_rows, min(block_width, rem_columns)));
		if (rem_columns > block_width) {
			rem_columns -= block_width;
		}
		
	}
	this->_current = 0;
	this->_block_size = block_width;
	this->_width = in_columns;
	this->_height = in_rows;
	return ERR_NONE;
}

// performs a swap-out of GPU memory. 
int GPU_Data::copy(int ref) {
	int error;
	ARRAY2D<char>* cpu = &(this->_data->at(this->_current));
	ARRAY2D<char>* gpu = this->_gpu;
	cudaMemcpy2D(cpu->data, gpu->pitch, gpu->data, cpu->pitch, cpu->width * sizeof(char), cpu->height, cudaMemcpyDeviceToHost);
	error = cudaGetLastError();
	cpu = &(this->_data->at(ref));
	cudaMemcpy2D(gpu->data, gpu->pitch, cpu->data, cpu->pitch, cpu->width * sizeof(char), cpu->height, cudaMemcpyHostToDevice);
	gpu->width = cpu->width;
	gpu->height = cpu->height;
	error = cudaGetLastError();
	this->_current = ref;
	if (error != cudaSuccess)
		return ERR_NONE;
	return error;
}

int GPU_Data::refresh() {
	int error;
	ARRAY2D<char>* cpu = &(this->_data->at(this->_current));
	ARRAY2D<char>* gpu = this->_gpu;
	cudaMemcpy2D(gpu->data, gpu->pitch, cpu->data, cpu->pitch, cpu->width*sizeof(char), cpu->height, cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		return ERR_NONE;
	return error;
}
std::string GPU_Data::debug() {
	std::stringstream st; 
	st << "GPU DATA,width="<<this->width()<<",height="<< this->height()<< ",pitch="<<this->gpu().pitch<<",blocksize="<< this->_block_size << ",chunks="<<this->_data->size()<<",current="<<this->_current << std::endl;
	return st.str();
}

__global__ void kernShift(char* array, char* tmpar, int pitch, int width, int height) {
	char tmp;
	int tid = (blockIdx.x *THREAD_SHIFT) + threadIdx.x;
	
	if (threadIdx.x < height) {
		tmp = REF2D(char,array,pitch, 0, tid);
		for (int i = 0; i < width-1; i++) {
			REF2D(char,array,pitch, i, tid) = REF2D(char,array,pitch, i+1, tid);
		}
		REF2D(char,array,pitch, width-1, tid) = tmp;
	}
}

void gpu_shift(GPU_Data& pack) {
	int per = (pack.gpu().height / THREAD_SHIFT) + ((pack.gpu().height % THREAD_SHIFT) > 0);
	char* tmpspace;
	cudaMalloc(&tmpspace, sizeof(char)*pack.gpu().height);
	kernShift<<<per,THREAD_SHIFT>>>(pack.gpu().data, tmpspace, pack.gpu().pitch,pack.gpu().width,pack.gpu().height);
	cudaDeviceSynchronize();
	assert(cudaGetLastError() == cudaSuccess);
}

void debugDataOutput(ARRAY2D<char> results, std::string outfile = "simdata.log") {
#ifndef NDEBUG
	char *lvalues;
	std::ofstream ofile(outfile.c_str());

	lvalues = (char*)malloc(results.height*results.pitch);
	cudaMemcpy2D(lvalues,results.pitch,results.data,results.pitch,results.width,results.height,cudaMemcpyDeviceToHost);
	for (unsigned int r = 0;r < results.width; r++) {
		for (unsigned int i = 0; i < results.height; i++) {
			char z = REF2D(char, lvalues, results.pitch, r, i);
			ofile << (int)z;
		}
		ofile << std::endl;
	}
	free(lvalues);
	ofile.close();
#endif
}
