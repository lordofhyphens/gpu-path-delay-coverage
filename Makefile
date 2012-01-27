CC=g++-4.4
CTAG_FLAGS=--langmap=C++:+.cu --append=yes
GPUCC=/opt/net/apps/cuda/bin/nvcc
header=simkernel.h markkernel.h coverkernel.h serial.h defines.h mergekernel.h ckt.h gpuckt.h gpudata.h vectors.h
logfile=log.txt
main=main.cc
tgenobj=Utility.o BlifParse.o Graph.o
src=ckt.cc node.cc vectors.cc
gsrc=gpuckt.cu gpudata.cu #simkernel.cu markkernel.cu  mergekernel.cu coverkernel.cu
obj=$(src:.cc=.o) $(main:.cc=.o)
gobj_cu=$(gsrc:.cu=.o)
gobj=$(gobj_cu:.c=.o)
out=fcount
CPFLAGS=-I/opt/net/apps/cuda/include -I/opt/net/apps/cudd/include -O2 -Wall -Werror # -DNDEBUG #-DNTIMING
CFLAGS=${CPFLAGS}
NVCFLAGS=-arch=sm_20 -O2 --compiler-options -I/opt/net/apps/cuda/include --compiler-options -Wall --compiler-options -Werror -ccbin ${CC} # --compiler-options -DNDEBUG --ptxas-options=-v #--compiler-options -DNTIMING  
PYLIB=_fsim.so
SWIGTEMPLATE=iscas.i sort.i gpuiscas.i simkernel.i

.PHONY: all
all: tags $(out)
.PHONY: pylib
pylib: ${PYLIB}
.PHONY: test
test: tags $(out)
	@./${out} data/c17.bench test.vec

.PHONY: cpu
cpu: CFLAGS = ${CPFLAGS} -DCPUCOMPILE
cpu: tags $(out)
%.o:
	@${CC} -c ${CFLAGS} $< -o $(@)
	
${out}: $(obj) ${gobj} 
	@${GPUCC} ${NVCFLAGS} -o ${out} ${obj} ${gobj}
${out}-cpu: $(obj) 
	${CC} -lrt -o ${out}-cpu ${obj} 
${obj}: ${src} ${header} ${main}
	@${CC} -c ${CFLAGS} $(@:.o=.cc)
${gobj}: ${gsrc}
	@${GPUCC} ${NVCFLAGS} -c $(@:.o=.cu)
tags: ${src} ${gsrc} ${header}
	ctags ${CTAG_FLAGS} $?
clean:
	rm -f ${out} ${out}-cpu ${obj} ${gobj} $(header:.h=.h.gch) ${logfile} ${gsrc:.cu=_wrap.cu} ${src:.cc=_wrap.cxx} ${PYLIB} ${gsrc:.cu=_wrap.o}  ${src:.cc=_wrap.o} swig_*.py

${src:.cc=_wrap.cxx}: ${SWIGTEMPLATE}
	swig -classic -c++ -python $(@:_wrap.cxx=.i)

${gsrc:.cu=_wrap.cu}: ${SWIGTEMPLATE}
	swig -classic -c++ -python -o $@ $(@:_wrap.cu=.i) 

${PYLIB}: ${SWIGPY} ${src} ${src:.cc=_wrap.cxx} ${gsrc:.cu=_wrap.cu} ${gsrc}
	${GPUCC} ${NVCFLAGS} -Xcompiler -fPIC -I/usr/include/python2.6 -c ${gsrc}
	${GPUCC} ${NVCFLAGS} -Xcompiler -fPIC -I/usr/include/python2.6 -c ${gsrc:.cu=_wrap.cu}
	${CC} -c ${src} -fPIC -O
	${CC} -c ${src:.cc=_wrap.cxx} -I/usr/include/python2.6 -fPIC -O
	${GPUCC} ${NVCGLAGS} -shared -Xcompiler -fPIC ${src:.cc=.o} ${src:.cc=_wrap.o} ${gsrc:.cu=_wrap.o} ${gsrc:.cu=.o} -o ${PYLIB}
