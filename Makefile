CC=gcc
src=main.c iscas.c
out=fcount
CFLAGS=-std=c99 -O2
all: ${src}
	${CC} ${CFLAGS} ${src} -o ${out}
clean:
	rm -f ${out}
