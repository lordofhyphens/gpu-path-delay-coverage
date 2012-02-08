#ifndef GPUCKT_H
#define GPUCKT_H

#include "ckt.h"
#include <utility>

typedef struct GPU_NODE_type {
	int type, nfi, nfo, po, offset,level; // initial offset for fanin and fanout
} GPUNODE;

// subclass of Circuit to provide GPU-friendly representation of the circuit.
class GPU_Circuit : public Circuit { 
	private: 
		int* _offset;
		GPUNODE* _gpu_graph;
		int _max_offset;
		int id(std::string) const;
	public:
		int* offset() const { return this->_offset;}
		int max_offset() const;
		GPUNODE* gpu_graph() const;
		void copy();
		~GPU_Circuit();
		GPU_Circuit();
};

template <class T> bool Yes(const T& item) {
	return true;
}
#endif //GPUCKT_H
