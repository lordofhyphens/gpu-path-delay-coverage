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

#define CHARPAIR (std::pair<char*,char*>())
typedef std::vector<ARRAY2D<char> >::iterator dataiter;

class GPU_Data : public CPU_Data {
	private:
		size_t _block_size;
		ARRAY2D<char>* _gpu; // fixed size GPU memory space.
		int copy(int);
	public: 
		ARRAY2D<char> gpu(int ref); // this will throw an out_of_range exception if ref > size; Also changes current.
		ARRAY2D<char> gpu() { return gpu(this->_current);}
		int refresh(); // ensures that the GPU memory space is equivalent to cpu-current.
		size_t block_width() { return this->_block_size;}
		int initialize(size_t, size_t, int);
		GPU_Data();
		GPU_Data(size_t rows, size_t columns);
		GPU_Data(size_t rows, size_t columns, int blockwidth);
		~GPU_Data();
		std::string debug();
		ARRAY2D<char> ar2d() const { return *(this->_gpu); }
};

void gpu_shift(GPU_Data& pack);
void debugDataOutput(ARRAY2D<char> results, std::string outfile);
#endif //GPUDATA_H
