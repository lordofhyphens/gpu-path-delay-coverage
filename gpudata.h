#ifndef GPUDATA_H
#define GPUDATA_H

#include <iostream> // included for debugging

#include <utility>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include "errors.h"
#include "defines.h"
#include "array2d.h"
#include "cpudata.h"


#define CHARPAIR (std::pair<uint8_t*,uint8_t*>())
typedef std::vector<ARRAY2D<uint8_t> >::iterator dataiter;

class GPU_Data : public CPU_Data {
	private:
		size_t _block_size;
		ARRAY2D<uint8_t>* _gpu; // fixed size GPU memory space.
		uint32_t copy(uint32_t);
	public: 
		ARRAY2D<uint8_t> gpu(uint32_t ref); // this will throw an out_of_range exception if ref > size; Also changes current.
		ARRAY2D<uint8_t> gpu() { return gpu(this->_current);}
		uint32_t refresh(); // ensures that the GPU memory space is equivalent to cpu-current.
		size_t block_width() { return this->_block_size;}
		uint32_t initialize(size_t, size_t, uint32_t);
		GPU_Data();
		GPU_Data(size_t rows, size_t columns);
		GPU_Data(size_t rows, size_t columns, uint32_t blockwidth);
		~GPU_Data();
		std::string debug();
		ARRAY2D<uint8_t> ar2d() const { return *(this->_gpu); }
};

void gpu_shift(GPU_Data& pack);
void debugDataOutput(ARRAY2D<uint8_t> results, std::string outfile);
#endif //GPUDATA_H
