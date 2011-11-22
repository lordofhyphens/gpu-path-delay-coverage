#ifndef DEFINES_H
#define DEFINES_H
#define N 32
//#define PATTERNS 4

#ifdef NDEBUG 
	#define DPRINT(...) 
#else
	#define DPRINT(...) fprintf(stderr, __VA_ARGS__)
#endif // DPRINT definition


#endif // include guard.
