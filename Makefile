CXX=g++-4.6
CUDA_DIR=/opt/net/apps/cuda-5.5
CTAG_FLAGS=--langmap=C++:+.cu --append=yes
GPCXX=${CUDA_DIR}/bin/nvcc
header= simkernel.h markkernel.h coverkernel.cuh mergekernel.cuh 
logfile=log.txt
main=main.cu
src=simkernel.cu markkernel.cu
obj=$(src:.cu=.o) $(main:.cc=.o)
out=fcount
CPFLAGS=-I${CUDA_DIR}/include -lrt -I./moderngpu/include -I/opt/net/apps/cudd/include -O2 -Wall -funsigned-char -funroll-loops -fopenmp #-Werror # -DNDEBUG #-DNTIMING
CFLAGS=${CPFLAGS}
NVCFLAGS=-g -G -arch=sm_20 --profile -O2 $(CPFLAGS:%=-Xcompiler %) -ccbin ${CXX} -Xptxas=-v # -Xcompiler -DNDEBUG - #-Xcompiler -DNTIMING  
PYLIB=_fsim.so

.PHONY: all util tags test
all: $(out) tags
test: $(out)
	cuda-memcheck ./fcount ../data/c17.level ../data/c17.vec

.SUFFIXES:
.SUFFIXES: .o .cu .cc

.cc.o: 
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<
.cu.o: util/defines.h
	$(GPCXX) -c -dc $(NVCFLAGS) -o $@ $<

tags: $(header) $(src) $(main) util/*
util:
	export CUDA_DIR="$(CUDA_DIR)" CFLAGS="$(CFLAGS)" &&  $(MAKE) -C util -e -j4 -w
	export CUDA_DIR="$(CUDA_DIR)" NVCFLAGS="$(NVCFLAGS)" && $(MAKE) -C util -e -j4 -w gpu

${out}: util $(obj)
	${GPCXX} $(NVCFLAGS) -o ${out} $(obj) util/*.o

clean:
	$(MAKE) -C util clean
	rm -rf *.o
