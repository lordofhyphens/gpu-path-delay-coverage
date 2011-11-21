#define N 32
#define PATTERNS 4
#ifdef NDEBUG 
	#define DPRINT(...) 
#else
	#define DPRINT(...) fprintf(stderr, __VA_ARGS__)
#endif
