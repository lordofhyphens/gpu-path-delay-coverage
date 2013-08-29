CXX=g++-4.6
CUDA_DIR=/opt/net/apps/cuda-5.5
CTAG_FLAGS=--langmap=C++:+.cu --append=yes
GPCXX=${CUDA_DIR}/bin/nvcc
header= simkernel.h markkernel.h coverkernel.h mergekernel.h 
logfile=log.txt
main=main.cc
src=simkernel.cu markkernel.cu mergekernel.cu coverkernel.cu gpu_hashmap.cu
obj=$(src:.cu=.o) $(main:.cc=.o)
out=fcount
CPFLAGS=-I${CUDA_DIR}/include -lrt -I/opt/net/apps/cudd/include -O2 -Wall -funsigned-char -fopenmp #-Werror # -DNDEBUG #-DNTIMING
CFLAGS=${CPFLAGS}
NVCFLAGS=-g -G -arch=sm_20 --profile -O2 $(CPFLAGS:%=-Xcompiler %) -ccbin ${CXX} -Xptxas=-v # -Xcompiler -DNDEBUG - #-Xcompiler -DNTIMING  
PYLIB=_fsim.so

.PHONY: all util tags
all: $(out) tags

.SUFFIXES:
.SUFFIXES: .o .cu .cc

.cc.o: 
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<
.cu.o:
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
