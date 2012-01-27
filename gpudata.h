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

#define CHARPAIR (std::pair<char*,char*>())
typedef std::vector<ARRAY2D<char> >::iterator dataiter;

class GPU_Data {
	private:
		std::vector<ARRAY2D<char> >* _data; // variable size CPU memory space.
		int _current; // current chunk on GPU.
		size_t _block_size;
		size_t _pitch;
		ARRAY2D<char>* _gpu; // fixed size GPU memory space.
		int initialize(size_t, size_t);
		int copy(int);
	public: 
		char* gpu(int ref); // this will throw an out_of_range exception if ref > size; Also changes current.
		char* gpu() { return gpu(this->_current);}
		char* cpu(int ref); // this will throw an out_of_range exception if ref > size
		char* cpu() { return cpu(this->_current);} // gets the CPU value for current;
		void gpu_refresh(); // ensures that the GPU memory space is equivalent to cpu-current.
		int current();
		size_t pitch() { return this->_pitch;}
		GPU_Data();
		GPU_Data(size_t rows, size_t columns);
		~GPU_Data();
		std::string debug();
};

#endif //GPUDATA_H
