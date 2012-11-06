CXX=g++-4.4
CTAG_FLAGS=--langmap=C++:+.cu --append=yes
GPCXX=/opt/net/apps/cuda/bin/nvcc
header= simkernel.h markkernel.h coverkernel.h mergekernel.h 
logfile=log.txt
main=main.cc
src=simkernel.cu markkernel.cu mergekernel.cu coverkernel.cu
obj=$(src:.cu=.o) $(main:.cc=.o)
out=fcount
CPFLAGS=-I/opt/net/apps/cuda/include -I/opt/net/apps/cudd/include -O2 -Wall -funsigned-char #-fopenmp -Werror # -DNDEBUG #-DNTIMING
CFLAGS=${CPFLAGS}
NVCFLAGS=-arch=sm_20 --profile -O3 -Xcompiler -I/opt/net/apps/cuda/include -Xcompiler -Wall -Xcompiler -Werror -ccbin ${CXX} -Xcompiler -funsigned-char -Xcompiler -fopenmp -Xptxas=-v # -Xcompiler -DNDEBUG - #-Xcompiler -DNTIMING  
PYLIB=_fsim.so

.PHONY: all
all: $(out)

.SUFFIXES:
.SUFFIXES: .o .cu .cc

.cc.o: 
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<
.cu.o:
	$(GPCXX) -c $(NVCFLAGS) -o $@ $<

util/*.o:
	export CFLAGS="$(CFLAGS)" &&  $(MAKE) -C util -e -j4 -w
	export NVCFLAGS="$(NVCFLAGS)" && $(MAKE) -C util -e -j4 -w gpu

${out}: $(obj) util/*.o 
	${GPCXX} $(NVCFLAGS) -o ${out} $+

clean:
	$(MAKE) -C util clean
	rm -rf *.o
