#ifndef DEFINES_H
#define DEFINES_H

#define UNINITIALIZED -1
#ifdef NDEBUG 
	#define DPRINT(...) 
#else
	#define DPRINT(...) fprintf(stderr, __VA_ARGS__)
#endif // DPRINT definition

#ifdef NDEBUG 
	#define GPRINT(A, ...) 
#else
	#define GPRINT(...) (if (tid == 0) { printf( __VA_ARGS__); })
#endif // GPRINT definition

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

// utility macros to makke addressing a little easier.
#define REF2D(TYPE,ARRAY,PITCH,X,Y) ( (((TYPE*)((char*)ARRAY + Y*PITCH))[X] ))
#define ADDR2D(TYPE,ARRAY,PITCH,X,Y) ( (((TYPE*)((char*)ARRAY + Y*PITCH))+X ))
//Usage: REF2D(int,o_count,p_count,pid,gid)
#define GREF(GRAPH,SUB,OFFSET, X) ( GRAPH[SUB[OFFSET+X]] )
#define FIN(AR, OFFSET, ID) ( AR[OFFSET+ID] ) 

#define TID (((blockIdx.y * blockDim.y) + threadIdx.x))
#define GID(OFFSET) (blockIdx.x + OFFSET)

//utility macro that serve same function as the stability lookup table.

#define STABLE(P, N) (3*(!P & N) + 2*(P & !N) + (P & N))

//thread-per-block sizes, per kernel.
#define SIM_BLOCK 768
#define MARK_BLOCK 128
#define MERGE_SIZE 512
#define COVER_BLOCK 256
#endif // include guard.
