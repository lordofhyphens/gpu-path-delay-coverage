#include "defines.h"
#include "errors.h"
#include "gpudata.h"
#include <cassert>
#include <string>
#include <sstream>
#include <fstream>
#include <utility>

std::pair<size_t, size_t> get_vector_dim(char* fvec);
int read_vectors(GPU_Data& pack, char* fvec, int chunksize);
