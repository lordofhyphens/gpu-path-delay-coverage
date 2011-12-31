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

#define NOT_IN(A) ( A >= T0 )
#define AND_OUT(A,B) ( (B>=T0)*(A!=S0) + (B==S1)*(A==S1)*2 + (B==S1)*(A>=T0) )
#define AND_IN(A,B) ((A==T0)*(B>S0)+(A==T1)*((B==S1)+(B==T1)))
#define OR_IN(A,B) ( (A>=T0)*((B==S0)+(B==T0))+(A==T1)*(B==T1) )
#define OR_OUT(A,B) ((A>=T0)*(B!=S1)+2*(B==S0)*(A==S0)+(A==S0)*(B>=T0)) 
#define XOR_IN(A,B) ( (A>=T0)*(B<=S1) )
#define XOR_OUT(A,B) ( (A>=T0)*(B<=S1) + (A<=S1)*(B>=T0) )

#endif // include guard.
