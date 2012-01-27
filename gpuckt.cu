#include "gpuckt.h"
#include <cuda.h>

int GPU_Circuit::max_offset() {
	return this->_max_offset;
}
GPUNODE* GPU_Circuit::gpu_graph() {
	return this->_gpu_graph;
}

// Copy the circuit representation to the GPU.
void GPU_Circuit::copy() {
	std::vector<NODEC>* g = this->graph;
	int *offsets;
	GPUNODE *ggraph = (GPUNODE*)malloc(sizeof(GPUNODE)*g->size());
	int off = 0;
	int maxoff = 0;
	for (std::vector<NODEC>::iterator i = g->begin(); i < g->end(); i++) {
		if (i->typ != UNKN) {
			maxoff += (i->nfi + i->nfo);
		}
	}
	offsets = (int*)malloc(sizeof(int)*maxoff); 
	for (unsigned int i = 0; i < g->size(); i++) {
		ggraph[i].type = graph->at(i).typ;
		ggraph[i].nfi = graph->at(i).nfi;
		ggraph[i].nfo = graph->at(i).nfo;
		ggraph[i].po = graph->at(i).po;
		ggraph[i].level = graph->at(i).level;
		if (graph->at(i).typ == 0)
			continue;
		ggraph[i].offset = off;
		// position of a particular node
		for (std::vector<std::string>::iterator fins = graph->at(i).fin.begin(); fins < graph->at(i).fin.end();fins++) {
			offsets[off] = this->id(*fins);
			off++;
		}
		for (std::vector<std::string>::iterator fots = graph->at(i).fot.begin(); fots < graph->at(i).fot.end();fots++) {
			offsets[off] = this->id(*fots);
			off++;
		}
	}
	this->_max_offset = off;
	cudaMalloc(&(this->_gpu_graph), sizeof(GPUNODE)*(g->size()));
	cudaMalloc(&(this->_offset), sizeof(int)*off);
	cudaMemcpy(this->_gpu_graph, ggraph, sizeof(GPUNODE)*(g->size()),cudaMemcpyHostToDevice);
	cudaMemcpy(this->_offset, offsets, sizeof(int)*off,cudaMemcpyHostToDevice);
}

int GPU_Circuit::id(std::string name) {
	return std::count_if(this->graph->begin(), find(this->graph->begin(),this->graph->end(),name), Yes<NODEC>);
}