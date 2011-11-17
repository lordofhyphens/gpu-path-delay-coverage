CC=gcc
GPUCC=nvcc
header=iscas.h
src=main.c iscas.c
gsrc=kernel.cu gpuiscas.cu
obj=$(src:.c=.o)
gobj=$(gsrc:.cu=.o)
out=fcount
CFLAGS=-std=c99 -O2
all: $(obj) ${gobj}
	${GPUCC} -o ${out} ${obj} ${gobj}
${obj}: ${src} ${header}
	${CC} -c ${CFLAGS} $^
${gobj}: ${gsrc}
	${GPUCC} -c $^
clean:
	rm -f ${out} ${obj} ${gobj} $(header:.h=.h.gch)
