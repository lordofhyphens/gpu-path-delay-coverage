#include <cuda.h>
#include "gpudata.h"
#include "defines.h"

GPU_Data::GPU_Data() {
	this->_gpu = new ARRAY2D<uint8_t>();
	this->_block_size = 0;
}
GPU_Data::GPU_Data(size_t rows, size_t columns) {
	this->_gpu = new ARRAY2D<uint8_t>();
	this->initialize(rows, columns, rows);
}
GPU_Data::GPU_Data(size_t rows, size_t columns, uint32_t blockwidth) {
	this->_gpu = new ARRAY2D<uint8_t>();
	this->initialize(rows, columns, blockwidth);
}
GPU_Data::~GPU_Data() {
	if (this->_gpu->data != NULL) {
		cudaFree(this->_gpu->data);
	}
}
ARRAY2D<uint8_t> GPU_Data::gpu(uint32_t ref) {
	if (ref == this->_current) {
		return *(this->_gpu);
	}
//	DPRINT("%s:%d - Switching to chunk %d\n", __FILE__, __LINE__, ref);
	uint32_t tmp = this->_current;
	uint32_t err; 
	try {
		err = this->copy(ref);
	} catch (std::out_of_range& oor) { 
		// handling the problem by returning NULL and ensuring that _current is not changed.
		this->_current = tmp;
		DPRINT("Out of range in swap.\n");
		return ARRAY2D<uint8_t>(NULL,0,0,0);
	}
	if (err != ERR_NONE) {
		DPRINT("Unknown error in swap.\n");
		return ARRAY2D<uint8_t>(NULL,0,0,0);
	}
	return *(this->_gpu);
}

// total size in columns, rows. 
uint32_t GPU_Data::initialize(size_t in_columns, size_t in_rows, uint32_t block_width) {
	uint32_t chunks = (in_columns / block_width) + ((in_columns % block_width) > 0);

	this->_gpu = new ARRAY2D<uint8_t>(NULL, in_rows, block_width, sizeof(uint8_t)*block_width);
	cudaMallocPitch(&(this->_gpu->data), &(this->_gpu->pitch), sizeof(uint8_t)*this->_gpu->width, in_rows);
	uint32_t rem_columns = in_columns;
	for (uint32_t i = 0; i < chunks;i++) {
		uint8_t* data = new uint8_t[in_rows*sizeof(uint8_t)*min(block_width,rem_columns)];
		this->_data->push_back(ARRAY2D<uint8_t>(data, in_rows, min(block_width, rem_columns),sizeof(uint8_t)*min(block_width,rem_columns)));
		assert(this->_data->back().data != NULL);
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
uint32_t GPU_Data::copy(uint32_t ref) {
	uint32_t error;
//	DPRINT("%s:%d - Copying chunk %d from GPU, %d to GPU.\n",__FILE__,__LINE__,_current, ref);
	ARRAY2D<uint8_t>* cpu = &(this->_data->at(this->_current));
	ARRAY2D<uint8_t>* gpu = this->_gpu;
//	DPRINT("%s:%d - Memcpy from GPU\n", __FILE__,__LINE__);
	cudaMemcpy2D(cpu->data, cpu->pitch, gpu->data, gpu->pitch, cpu->width * sizeof(uint8_t), cpu->height, cudaMemcpyDeviceToHost);
//	DPRINT("%s:%d - Memcpy from GPU\n", __FILE__,__LINE__);
	error = cudaGetLastError();
//	DPRINT("%s:%d - Getting reference to CPU", __FILE__,__LINE__);
	cpu = &(this->_data->at(ref));
	cudaMemcpy2D(gpu->data, gpu->pitch, cpu->data, cpu->pitch, cpu->width * sizeof(uint8_t), cpu->height, cudaMemcpyHostToDevice);
	gpu->width = cpu->width;
	gpu->height = cpu->height;
	error = cudaGetLastError();
	this->_current = ref;
//	DPRINT("%s:%d - Finished copy.\n", __FILE__,__LINE__);
	if (error != cudaSuccess)
		return ERR_NONE;
	return error;
}

uint32_t GPU_Data::refresh() {
	uint32_t error;
	ARRAY2D<uint8_t>* cpu = &(this->_data->at(this->_current));
	ARRAY2D<uint8_t>* gpu = this->_gpu;
	cudaMemcpy2D(gpu->data, gpu->pitch, cpu->data, cpu->pitch, cpu->width*sizeof(uint8_t), cpu->height, cudaMemcpyHostToDevice);
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

__global__ void kernShift(uint8_t* array, uint8_t* tmpar, uint32_t pitch, uint32_t width, uint32_t height) {
	uint8_t tmp;
	uint32_t tid = (blockIdx.x *THREAD_SHIFT) + threadIdx.x;
	
	if (threadIdx.x < height) {
		tmp = REF2D(uint8_t,array,pitch, 0, tid);
		for (uint32_t i = 0; i < width-1; i++) {
			REF2D(uint8_t,array,pitch, i, tid) = REF2D(uint8_t,array,pitch, i+1, tid);
		}
		REF2D(uint8_t,array,pitch, width-1, tid) = tmp;
	}
}

void gpu_shift(GPU_Data& pack) {
	uint32_t per = (pack.gpu().height / THREAD_SHIFT) + ((pack.gpu().height % THREAD_SHIFT) > 0);
	uint8_t* tmpspace;
	cudaMalloc(&tmpspace, sizeof(uint8_t)*pack.gpu().height);
	kernShift<<<per,THREAD_SHIFT>>>(pack.gpu().data, tmpspace, pack.gpu().pitch,pack.gpu().width,pack.gpu().height);
	cudaDeviceSynchronize();
	assert(cudaGetLastError() == cudaSuccess);
}

void debugDataOutput(ARRAY2D<uint8_t> results, std::string outfile = "simdata.log") {
#ifndef NDEBUG
	uint8_t *lvalues;
	std::ofstream ofile(outfile.c_str());

	lvalues = (uint8_t*)malloc(results.height*results.pitch);
	cudaMemcpy2D(lvalues,results.pitch,results.data,results.pitch,results.width,results.height,cudaMemcpyDeviceToHost);
	for (uint32_t r = 0;r < results.width; r++) {
		for (uint32_t i = 0; i < results.height; i++) {
			uint8_t z = REF2D(uint8_t, lvalues, results.pitch, r, i);
			ofile << (uint32_t)z;
		}
		ofile << std::endl;
	}
	free(lvalues);
	ofile.close();
#endif
}
