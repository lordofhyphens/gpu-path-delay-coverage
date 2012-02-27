#ifndef ARRAY2D_H
#define ARRAY2D_H
#include <cassert>
#include <cstdlib>
template <class t>
struct ARRAY2D {
	t* data;
	size_t height;
	size_t pitch;
	size_t width;
	ARRAY2D(t *, int, int);
	ARRAY2D();
	ARRAY2D(t *, int, int, int);
	ARRAY2D(size_t, size_t, size_t);
	ARRAY2D(const ARRAY2D<t>&);
	ARRAY2D(size_t height, size_t width);
	void initialize(t*, size_t, size_t, size_t);
	size_t size();
	size_t bwidth();
	size_t mem_footprint;
};

template <class t>
ARRAY2D<t>::ARRAY2D(t *in, int height, int width) {
	this->initialize(in, height, width, width*sizeof(t));
}
template <class t>
ARRAY2D<t>::ARRAY2D(size_t height, size_t width, size_t pitch) {
	t* in = new t[pitch*height];
	assert(in != NULL);
	this->initialize(in, height, width, pitch);
}
template <class t>
ARRAY2D<t>::ARRAY2D(size_t height, size_t width) {
	t* in = new t[width*height];
	assert(in != NULL);
	this->initialize(in, height, width, sizeof(t)*width);
}

template <class t>
ARRAY2D<t>::ARRAY2D(const ARRAY2D<t>& other) {
	this->data = other.data;
	this->height = other.height;
	this->pitch = other.pitch;
	this->width = other.width;
	this->mem_footprint = other.mem_footprint;
}
template <class t> 
void ARRAY2D<t>::initialize(t *in, size_t height, size_t width, size_t pitch) {
	this->data = in;
	this->pitch = pitch;
	this->height = height;
	this->width = width;
	this->mem_footprint = height * pitch;
}
template <class t>
ARRAY2D<t>::ARRAY2D() {
	this->initialize(NULL, 0, 0, 0);
}
template <class t>
ARRAY2D<t>::ARRAY2D(t *in, int height, int width, int pitch) {
	this->initialize(in, height, width, pitch);
}
template <class t>
size_t ARRAY2D<t>::size() {
	return (sizeof(t) * height * pitch);
}
template <class t>
size_t ARRAY2D<t>::bwidth() {
	return (sizeof(t) * width);
}
#endif
