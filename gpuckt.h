#ifndef GPUCKT_H
#define GPUCKT_H

#include "ckt.h"

typedef struct GPU_NODE_type {
	int type, nfi, nfo, po, offset,level; // initial offset for fanin and fanout
} GPUNODE;

// subclass of Circuit to provide GPU-friendly representation of the circuit.
class GPU_Circuit : public Circuit { 
	private: 
		int* _offset;
		GPUNODE* _gpu_graph;
		int _max_offset;
		int id(std::string);
	public:
		int* offset() { return this->_offset;}
		int max_offset();
		GPUNODE* gpu_graph();
		void copy();
};

template <class T> bool Yes(const T& item) {
	return true;
}
#endif //GPUCKT_H
