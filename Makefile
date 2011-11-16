CC=gcc
GPUCC=nvcc
src=main.c iscas.c
gsrc=kernel.cu
obj=$(src:.c=.o)
gobj=$(gsrc:.cu=.o)
gpusrc=main.cu
out=fcount
CFLAGS=-std=c99 -O2
all: $(obj) ${gobj}
	${GPUCC} -o ${out} ${obj} ${gobj}
${obj}: ${src}
	${CC} -c ${CFLAGS} $^
${gobj}: ${gsrc}
	${GPUCC} -c ${gsrc} $^
clean:
	rm -f ${out} ${obj}
