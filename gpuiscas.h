#ifndef GPUISCAS_H
#define GPUISCAS_H
#include <cuda.h>
#include "iscas.h"
int* gpuLoadVectors(int** input, size_t width, size_t height);
int* gpuLoad1DVector(int** input, size_t width, size_t height);

int* gpuLoadFans(int* offset, int maxid);
GPUNODE* gpuLoadCircuit(const GPUNODE* graph, int maxid);
LINE* gpuLoadLines(LINE* graph, int maxid);
template <class t>
struct ARRAY2D {
	t* data;
	int height;
	int width;
	ARRAY2D(t *, int, int);
	size_t size();
	size_t bwidth();
};
template <class t>
ARRAY2D<t>::ARRAY2D(t *in, int height, int width) {
	this->data = in;
	this->height = height;
	this->width = width;
}
template <class t>
size_t ARRAY2D<t>::size() {
	return (sizeof(t) * height * width);
}
template <class t>
size_t ARRAY2D<t>::bwidth() {
	return (sizeof(t) * width);
}
#endif
