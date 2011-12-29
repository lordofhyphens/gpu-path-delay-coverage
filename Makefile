CC=g++-4.4
CTAG_FLAGS=--langmap=C++:+.cu --append=yes
GPUCC=/opt/net/apps/cuda/bin/nvcc
header=iscas.h gpuiscas.h simkernel.h markkernel.h coverkernel.h sort.h serial.h defines.h
logfile=log.txt
main=main.cc
tgenobj=Utility.o BlifParse.o Graph.o
src=iscas.cc sort.cc serial.cc
gsrc=gpuiscas.cu simkernel.cu markkernel.cu coverkernel.cu
obj=$(src:.cc=.o) $(main:.cc=.o)
gobj_cu=$(gsrc:.cu=.o)
gobj=$(gobj_cu:.c=.o)
out=fcount
CFLAGS=-I/opt/net/apps/cuda/include -I/opt/net/apps/cudd/include -O -Wall -Werror
NVCFLAGS=-arch=sm_20 -O --compiler-options -I/opt/net/apps/cuda/include --compiler-options -Wall --compiler-options -Werror -ccbin ${CC} 
PYLIB=_fsim.so
SWIGTEMPLATE=iscas.i sort.i gpuiscas.i simkernel.i

.PHONY: all
all: tags $(out)
.PHONY: pylib
pylib: ${PYLIB}

test: tags $(out)
	@./${out} data/c17.isc data/c17.vec 2> ${logfile}
	@egrep -e "Total" -e "time " -e "Vector [0-9]{1,2}:" -e "Line:" ${logfile} | tail -n60

.PHONY: cpu
cpu: CFLAGS = -DCPUCOMPILE -g
cpu: tags $(out)-cpu

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
	${CC} -c ${src} -fPIC -O2
	${CC} -c ${src:.cc=_wrap.cxx} -I/usr/include/python2.6 -fPIC -O2
	${GPUCC} ${NVCGLAGS} -shared -Xcompiler -fPIC ${src:.cc=.o} ${src:.cc=_wrap.o} ${gsrc:.cu=_wrap.o} ${gsrc:.cu=.o} -o ${PYLIB}
