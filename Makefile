CC=gcc-4.4
GPUCC=nvcc
header=iscas.h
src=main.c iscas.c
gsrc=kernel.cu gpuiscas.cu
obj=$(src:.c=.o)
gobj=$(gsrc:.cu=.o)
out=fcount
CFLAGS=-std=c99 -O2
LIB=-lcuda
all: tags $(out) $(out)-cpu
cpu: tags $(out)-cpu

${out}: $(obj) ${gobj} 
	${GPUCC} -o ${out} ${obj} ${gobj}
${out}-cpu: $(obj)
	${CC} -o ${out}-cpu ${obj}

${obj}: ${src} ${header}
	${CC} -c ${CFLAGS} $^
${gobj}: ${gsrc}
	${GPUCC} -ccbin g++-4.4 -c $^
tags: ${src} ${gsrc} ${header}
	ctags ${src} ${gsrc}
clean:
	rm -f ${out} ${obj}-cpu ${obj} ${gobj} $(header:.h=.h.gch)
