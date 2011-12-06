#ifndef DEFINES_H
#define DEFINES_H

#define UNINITIALIZED -1
#ifdef NDEBUG 
	#define DPRINT(...) 
#else
	#define DPRINT(...) fprintf(stderr, __VA_ARGS__)
#endif // DPRINT definition

#ifdef NTIMING
	#define TPRINT(...) 
#else
	#define TPRINT(...) fprintf(stderr, __VA_ARGS__)
#endif // TPRINT definition


#endif // include guard.
