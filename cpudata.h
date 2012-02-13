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
#include <stdexcept>
#include <algorithm>
class CPU_Data {
    protected:
		std::vector<ARRAY2D<char> >* _data; // variable size CPU memory space.
		size_t _width, _height;
		int _current; // current chunk on GPU.
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
		ARRAY2D<char> cpu(int ref); // this will throw an out_of_range exception if ref > size
		ARRAY2D<char> cpu() { return cpu(this->_current);} // gets the CPU value for current;
};

#endif // CPUDATA_H
