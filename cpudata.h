#ifndef CPUDATA_H
#define CPUDATA_H

#include "array2d.h"
#include "errors.h"
#include "defines.h"
#include "array2d.h"
#include <iostream> // included for debugging
#include <utility>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <string>
class CPU_Data {
    protected:
		std::vector<ARRAY2D<uint8_t> >* _data; // variable size CPU memory space.
		size_t _width, _height;
		uint32_t _current; // current chunk on GPU.
    public:
		size_t height() { return this->_height;}
		size_t width() { return this->_width;}
		size_t size() { return this->_data->size();}
		int initialize(size_t, size_t);
        CPU_Data();
        CPU_Data(size_t, size_t);
        ~CPU_Data();
        std::string debug();
		int current();
		std::string print(); 
		ARRAY2D<uint8_t> cpu(int ref); // this will throw an out_of_range exception if ref > size
		ARRAY2D<uint8_t> cpu() { return cpu(this->_current);} // gets the CPU value for current;
		void save(const char *file);
		void load(const char *file);
};

#endif // CPUDATA_H
