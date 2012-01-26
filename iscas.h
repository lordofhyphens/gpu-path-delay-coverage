#ifndef ISCAS_H
#define ISCAS_H
/*************************************************************
Header Files
*************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <assert.h>
#include <limits.h>
/*************************************************************
Constant Declarations 
*************************************************************/
// VARIOUS CONSTANTS
#define Mfnam      20			// max size for a file name
#define Mnod   305000 		        // max number of nodes in a graph/node
#define Mlin      512			// max size of characters in a line
#define Mnam       25			// max size of a node name
#define Mtyp       10			// max type of nodes/gates
#define Mout       16		        // max node out degree (nfo)
#define Min         9			// max node in degree (nfi)
#define Mpi       233			// max number of primary inputs
#define Mpo       140			// max number of primary outputs
#define Mpt       10			// max number of input patterns in .vec file
#define Mft       10			// max number of stuck at faults in .faults file

// NODE TYPE CONSTANTS 
#define INPT 1				// Primary Input
#define AND  2				// AND 
#define NAND 3				// NAND 
#define OR   4				// OR 
#define NOR  5				// NOR 
#define XOR  6				// XOR 
#define XNOR 7				// XNOR 
#define BUFF 8				// BUFFER 
#define NOT  9				// INVERTER 
#define FROM 10				// STEM BRANCH

// Signal type constants
#define S0 0
#define S1 1
#define T0 2 // 1->0 transition, or "Transition to 0"
#define T1 3 // 0->1 transition, or "Transition to 1"
/*************************************************************
Structure Declarations 
*************************************************************/
//1.Stucture declaration for LIST
typedef struct LIST_type {
	int  id;                //id for current element
	int line;               //Value of this output
	struct LIST_type *nxt;  //pointer to next id element
	                        //( if there is no element, then it will be NULL)		
} LIST;
//2.Stucture declaration for NODE
typedef struct NODE_type
{
	char nam[Mnam];         //name of the node
	int typ,nfi,nfo,po;     //type, nooffanins, nooffanouts,primaryo/p
	int level;
	int mar,val,fval;       //marker,correctvalue,faultvalue
	LIST *fin,*fot;         //fanin members, fanout members 
	NODE_type(const NODE_type& n);
} NODE;
typedef struct LINE_type {
	int logic; // logic value for this line.
	int prev, next; // preceding gate, next gate
} LINE;

typedef struct GPU_NODE_type {
	int type, nfi, nfo, po, offset,level; // initial offset for fanin and fanout
} GPUNODE;

typedef struct GPUNODE_INFO_type {
	int max_offset;
	GPUNODE* graph;
	int* offsets;
} GPUNODE_INFO;
//3.Stucture declaration for PATTERN
typedef struct PATTERN_type
{
 char piv[Mpi];    //primary input vector(size is not declared)
} PATTERN;
//4.Stucture declaration for FAULT
typedef struct FAULT_type
{
 int nod,sval;      //stuckatnode,stuckatvalue
} FAULT;
/*************************************************************************
Functions in iscas.c
**************************************************************************/
/*************************************************************************
LIST Structure Functions
**************************************************************************/
void InsertList(LIST **,int, int);
void PrintList(LIST *);
void FreeList(LIST **);
/*************************************************************************
 NODE Structure Functions
**************************************************************************/
int ReadIsc(FILE *,NODE *);
void InitializeCircuit(NODE *,int);
void InitializeLines(LINE *,int);
int AssignType(char *);
void PrintCircuit(NODE *,int);
int EnumerateLines(NODE *graph, LINE *lgraph,int);
GPUNODE_INFO GraphsetToArrays(NODE* graph, LINE* lgraph, int maxid);
void PrintLines(LINE* lgraph, int lcnt); 
int verifyArrays(GPUNODE_INFO info, LINE* lgraph, int maxid);
int readVectors(int** vecs, FILE* fvec);
timespec diff(timespec start, timespec end);
float floattime(timespec time);
#endif
