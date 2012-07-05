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
		uint32_t* _offset;
		GPUNODE* _gpu_graph;
		uint32_t _max_offset;
		uint32_t id(std::string) const;
	public:
		uint32_t* offset() const { return this->_offset;}
		uint32_t max_offset() const;
		GPUNODE* gpu_graph() const;
		void copy();
		~GPU_Circuit();
		GPU_Circuit();
};

template <class T> bool Yes(const T& item) {
	return true;
}
#endif //GPUCKT_H
