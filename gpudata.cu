#include <cuda.h>
#include "gpudata.h"

static void delete_data(const ARRAY2D<char>& item) {
	if (item.data != NULL)
		free(item.data);
}

GPU_Data::GPU_Data() {
	this->_data = new std::vector<ARRAY2D<char> >();
	this->_gpu = new ARRAY2D<char>();
	this->_block_size = 0;
}
GPU_Data::GPU_Data(size_t rows, size_t columns) {
	this->_data = new std::vector<ARRAY2D<char> >();
	this->_gpu = new ARRAY2D<char>();
	this->initialize(rows, columns);
}
GPU_Data::~GPU_Data() {
	std::for_each(this->_data->begin(), this->_data->end(), delete_data);
	if (this->_gpu->data != NULL)
		cudaFree(this->_gpu->data);
}
char* GPU_Data::gpu(int ref) {
	if (ref == this->_current) {
		return this->_gpu->data;
	}
	int tmp = this->_current;
	int err; 
	try {
		err = this->copy(ref);
	} catch (std::out_of_range& oor) { 
		// handling the problem by returning NULL and ensuring that _current is not changed.
		this->_current = tmp;
		return NULL;
	}
	if (err != ERR_NONE)
		return NULL;
	return this->_gpu->data;
}

char* GPU_Data::cpu(int ref) {
	return this->_data->at(ref).data;
}
// total size in columns, rows. 
int GPU_Data::initialize(size_t in_columns, size_t in_rows) {
	size_t free_memory, total_memory, free_cols;
	cudaMemGetInfo(&free_memory, &total_memory);
	free_cols = free_memory / (sizeof(char)*in_rows);
	free_cols = (free_cols < sizeof(char)*in_columns ? free_cols : (sizeof(char)*in_columns + (WARP_SIZE - (sizeof(char)*in_columns % WARP_SIZE)))); // set to the smaller of the two
	std::cout << "Warp size: " << WARP_SIZE << " and free_cols " << free_cols  << " " << free_cols % WARP_SIZE << std::endl;
	// cut back to the nearest warp size
	free_cols = (free_cols % WARP_SIZE > 0 ? free_cols : free_cols - (free_cols % WARP_SIZE));
	// Allocate on the GPU.
	delete this->_gpu;
	this->_gpu = new ARRAY2D<char>(NULL, in_rows, free_cols, free_cols);
	cudaMallocPitch(&(this->_gpu->data), &(this->_gpu->pitch), this->_gpu->bwidth(), in_rows);
	this->_pitch = this->_gpu->pitch;
	int chunks = (in_columns / free_cols) + ((in_columns % free_cols) > 0);
	for (int i = 0; i < chunks;i++) {
		this->_data->push_back(ARRAY2D<char>(in_rows, free_cols, this->_pitch));
	}
	this->_current = 0;
	this->_block_size = this->_gpu->width;
	return ERR_NONE;
}

// performs a swap-out of GPU memory. 
int GPU_Data::copy(int ref) {
	int error;
	ARRAY2D<char>* cpu = &(this->_data->at(this->_current));
	cudaMemcpy2D(cpu->data, this->_pitch, this->_gpu->data, this->_pitch, cpu->bwidth(), cpu->height, cudaMemcpyDeviceToHost);
	error = cudaGetLastError();
	cudaMemcpy2D(this->_gpu->data, this->_pitch, cpu->data, this->_pitch, cpu->bwidth(), cpu->height, cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	this->_current = ref;
	if (error != cudaSuccess)
		return ERR_NONE;
	return error;
}

int GPU_Data::current() {
	return this->_current;
}

std::string GPU_Data::debug() {
	std::stringstream st; 
	st << "GPU DATA,blocksize="<< this->_block_size << ",chunks="<<this->_data->size()<<",current="<<this->_current << std::endl;
	return st.str();
}
