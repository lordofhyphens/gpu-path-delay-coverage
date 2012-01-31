#include <cuda.h>
#include "gpudata.h"

GPU_Data::GPU_Data() {
	this->_gpu = new ARRAY2D<char>();
	this->_block_size = 0;
}
GPU_Data::GPU_Data(size_t rows, size_t columns) {
	this->_gpu = new ARRAY2D<char>();
	this->initialize(rows, columns);
}
GPU_Data::~GPU_Data() {
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

// total size in columns, rows. 
int GPU_Data::initialize(size_t in_columns, size_t in_rows) {
	size_t free_memory, total_memory, free_cols;
	bool undersized = false;
	cudaMemGetInfo(&free_memory, &total_memory);
	free_cols = free_memory / (sizeof(char)*in_rows);
	undersized = (free_cols < sizeof(char)*in_columns ? false : true);
	free_cols = (free_cols < sizeof(char)*in_columns ? free_cols : (sizeof(char)*in_columns + (WARP_SIZE - (sizeof(char)*in_columns % WARP_SIZE)))); // set to the smaller of the two
//	std::cout << "Warp size: " << WARP_SIZE << " and free_cols " << free_cols  << " " << free_cols % WARP_SIZE << std::endl;
	// cut back to the nearest warp size
	free_cols = (free_cols % WARP_SIZE > 0 ? free_cols : free_cols - (free_cols % WARP_SIZE));
	// Allocate on the GPU.
	delete this->_gpu;
	if (undersized == true) {
		this->_gpu = new ARRAY2D<char>(NULL, in_rows, in_columns, free_cols);
	} else { 
		this->_gpu = new ARRAY2D<char>(NULL, in_rows, free_cols, free_cols);
	}
	cudaMallocPitch(&(this->_gpu->data), &(this->_gpu->pitch), this->_gpu->bwidth(), in_rows);
	this->_pitch = this->_gpu->pitch;
	int chunks = (in_columns / free_cols) + ((in_columns % free_cols) > 0);
	for (int i = 0; i < chunks;i++) {
		this->_data->push_back(ARRAY2D<char>(in_rows, free_cols, this->_pitch));
	}
	this->_current = 0;
	this->_block_size = this->_gpu->width;
	this->_width = in_columns;
	this->_height = in_rows;
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

int GPU_Data::refresh() {
	int error;
	ARRAY2D<char>* cpu = &(this->_data->at(this->_current));
	cudaMemcpy2D(this->_gpu->data, this->_pitch, cpu->data, this->_pitch, cpu->bwidth(), cpu->height, cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		return ERR_NONE;
	return error;
}
std::string GPU_Data::debug() {
	std::stringstream st; 
	st << "GPU DATA,width="<<this->width()<<",height="<< this->height()<< ",pitch="<<this->pitch()<<",blocksize="<< this->_block_size << ",chunks="<<this->_data->size()<<",current="<<this->_current << std::endl;
	return st.str();
}

__global__ void kernShift(char* array, char* tmpar, int pitch, int cycle, int width) {
	char tmp;
	int first = width-(cycle*threadIdx.x);
	if (first >= 0)
		tmp = REF2D(char,array,pitch, first, blockIdx.x);
	if (threadIdx.x == 0) {
		tmp = tmpar[blockIdx.x];
	}
	__syncthreads();
	if (threadIdx.x == THREAD_SHIFT-1) {
		tmpar[blockIdx.x] = tmp;
	} else {
		if (first > 0) {
			REF2D(char,array,pitch, first-1, blockIdx.x) = tmp;
		} else if (first == 0) {
			REF2D(char,array,pitch, width, blockIdx.x) = tmp;
		}
	}
	__syncthreads();
}

void gpu_shift(GPU_Data& pack) {
	int per = (pack.width() / THREAD_SHIFT) + (pack.width() % THREAD_SHIFT > 0);
	char* tmpspace;
	cudaMalloc(&tmpspace, sizeof(char)*pack.height());
	for (int i = 0; i < per; i++) {
		kernShift<<<pack.height(), THREAD_SHIFT>>>(pack.gpu(), tmpspace, pack.pitch(),i,pack.width());
		cudaDeviceSynchronize();
		assert(cudaGetLastError() == cudaSuccess);
	}
}


