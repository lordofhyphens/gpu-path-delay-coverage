CC=g++-4.4
GPUCC=nvcc
header=iscas.h gpuiscas.h kernel.h
src=main.cc iscas.cc
gsrc=kernel.cu gpuiscas.cu
obj=$(src:.cc=.o)
gobj_cu=$(gsrc:.cu=.o)
gobj=$(gobj_cu:.c=.o)
out=fcount
CFLAGS=-I/opt/net/apps/cuda/include -O2
NVCFLAGS=-arch=sm_20 -O2
LIB=-lcuda
all: tags $(out)

test: tags $(out)
	./${out} data/c17.isc

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
	ctags ${src} ${gsrc}
clean:
	rm -f ${out} ${out}-cpu ${obj} ${gobj} $(header:.h=.h.gch)
