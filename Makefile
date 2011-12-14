CC=g++-4.4
CTAG_FLAGS=--langmap=C++:+.cu
GPUCC=/opt/net/apps/cuda/bin/nvcc
header=iscas.h gpuiscas.h simkernel.h markkernel.h coverkernel.h sort.h
logfile=log.txt
src=main.cc iscas.cc sort.cc
gsrc=gpuiscas.cu simkernel.cu markkernel.cu coverkernel.cu
obj=$(src:.cc=.o)
gobj_cu=$(gsrc:.cu=.o)
gobj=$(gobj_cu:.c=.o)
out=fcount
CFLAGS=-I/opt/net/apps/cuda/include -g
NVCFLAGS=-arch=sm_20 -g
LIB=-lcuda
all: tags $(out)

test: tags $(out)
	@./${out} data/c17.isc data/c17.vec 2> ${logfile}
	@egrep -e "Total" -e "time " -e "Vector [0-9]{1,2}:" -e "Line:" ${logfile} | tail -n60

cpu: tags $(out)-cpu

${out}: $(obj) ${gobj} 
	${GPUCC} ${NVCFLAGS} -o ${out} ${obj} ${gobj}
${out}-cpu: $(obj)
	${CC} -cuda -o ${out}-cpu ${obj} ${gobj}

${obj}: ${src} ${header}
	${CC} -c ${CFLAGS} $^
${gobj}: ${gsrc}
	${GPUCC} ${NVCFLAGS} -ccbin g++-4.4 -c $^
tags: ${src} ${gsrc} ${header}
	ctags ${CTAG_FLAGS} ${src} ${gsrc}
clean:
	rm -f ${out} ${out}-cpu ${obj} ${gobj} $(header:.h=.h.gch) ${logfile}
