#ifndef ARRAY2D_H
#define ARRAY2D_H
template <class t>
struct ARRAY2D {
	t* data;
	size_t height;
	size_t pitch;
	size_t width;
	ARRAY2D(t *, int, int);
	ARRAY2D();
	ARRAY2D(t *, int, int, int);
	size_t size();
	size_t bwidth();
	size_t mem_footprint;
};

template <class t>
ARRAY2D<t>::ARRAY2D(t *in, int height, int width) {
	this->data = in;
	this->pitch = width*sizeof(t);
	this->height = height;
	this->width = width;
	this->mem_footprint = height * pitch;
}

template <class t>
ARRAY2D<t>::ARRAY2D() {
	this->data = NULL;
	this->pitch = 0;
	this->height = 0;
	this->width = 0;
	this->mem_footprint = sizeof(t) * height * pitch;
}
template <class t>
ARRAY2D<t>::ARRAY2D(t *in, int height, int width, int pitch) {
	this->data = in;
	this->pitch = pitch;
	this->height = height;
	this->width = width;
	this->mem_footprint = sizeof(t) * height * pitch;
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
