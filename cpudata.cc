#include "cpudata.h"
/*
static void delete_data(const ARRAY2D<uint8_t>& item) {
	delete item.data;
}
*/
CPU_Data::CPU_Data() {
	this->_data = new std::vector<ARRAY2D<uint8_t> >();
    this->_width = 0;
    this->_height =0;
}
CPU_Data::CPU_Data(size_t rows, size_t columns) {
	this->_data = new std::vector<ARRAY2D<uint8_t> >();
	this->initialize(rows, columns);
}

CPU_Data::~CPU_Data() {
//	std::for_each(this->_data->begin(), this->_data->end(), delete_data);
}

ARRAY2D<uint8_t> CPU_Data::cpu(int ref) {
	return this->_data->at(ref);
}

int CPU_Data::initialize(size_t in_columns, size_t in_rows) {
    this->_data->push_back(ARRAY2D<uint8_t>(in_rows, in_columns, sizeof(uint8_t)*in_columns));
	this->_current = 0;
    this->_width = in_rows;
	this->_height = in_columns;
    return ERR_NONE;
}
std::string CPU_Data::print() {
	std::stringstream st; 
	for (unsigned int r = 0;r < this->cpu().width; r++) {
		for (unsigned int i = 0; i < this->cpu().height; i++) {
			uint8_t z = REF2D(uint8_t, this->cpu().data, this->cpu().pitch, r, i);
			st << (int)z;
		}
		st << std::endl;
	}
	return st.str();
}
std::string CPU_Data::debug() {
	std::stringstream st; 
	st << "CPU DATA,width="<<this->width()<<",height="<< this->height()<< ",chunks="<<this->_data->size()<<",current="<<this->_current << std::endl;
	return st.str();
}

void CPU_Data::save(const char *file) {
	std::fstream filestr;
	filestr.open (file, std::fstream::out);
	filestr << this->_data->size();
	for (unsigned j = 0; j < _data->size(); j++) {
		filestr << this->cpu(j).height << this->cpu(j).width << this->cpu(j).pitch;

		for (unsigned int i = 0; i < this->cpu(j).height * this->cpu(j).pitch;j++) {
			filestr << this->cpu(j).data[i];
		}
	}
	filestr.close();
}

void CPU_Data::load(const char *file) {
	std::fstream filestr;
	unsigned int size = 0;
	filestr >> size;
	int height, width, pitch;
	ARRAY2D<uint8_t>* data;
	for (unsigned int j = 0; j < size; j++) {
		filestr.open (file, std::fstream::in);
		filestr >> height;
		filestr >> width;
		filestr >> pitch;
		data = new ARRAY2D<uint8_t>(height, width, pitch);
		for (int i = 0; i < height * pitch; i++) {
			filestr >> data->data[i];
		}
		this->_data->push_back(*data);
		delete data;
	}
	filestr.close();
}
