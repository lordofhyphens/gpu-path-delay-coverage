
#include <stdio.h>
#include <stdbool.h>

#include "kernel.h"
#include "iscas.h"
#include "gpuiscas.h"

#include "defines.h"
int procgate(NODE gnode, int settle); 
int main(int argc, char ** argv) {
	FILE *fisc, *fvec;
	int *dres, *fans;
	GPUNODE *dgraph;
	LINE *dlines;
	int** res;
	NODE graph[Mnod];
	LINE lgraph[Mnod];
	GPUNODE_INFO test;
	int lcnt, ncnt; // count of lines in the circuit
	int PATTERNS = 4;

	// Load circuit from file
	fisc=fopen(argv[1],"r");
	ncnt = ReadIsc(fisc,graph);
	for (int i = 0; i < Mnod; i++)
		InitializeLines(lgraph, i);
	lcnt = EnumerateLines(graph,lgraph,ncnt);

	test = GraphsetToArrays(graph, lgraph, ncnt);

	DPRINT("I\tLineID\tPrev\tNext\n");
	for(int i = 0; i < test.max_offset; i++) {
		DPRINT(" %d:\t%d\t%d\t%d\n",i,test.offsets[i],lgraph[test.offsets[i]].prev,lgraph[test.offsets[i]].next );
	}
	DPRINT("ID:\tType\n");
	for(int i = 0; i < ncnt; i++) {
		DPRINT(" %d:\t%d\n",i,test.graph[i].type);
	}
	int **vec = (int**)malloc(sizeof(int*)*4);
	int vecA[5] = {1,0,0,1,1}; //v0
	int vecB[5] = {0,1,0,0,0}; //v1
	int vecC[5] = {1,0,1,0,1}; //v2
	int vecD[5] = {0,0,1,1,1}; //v3
	vec[0] = vecA;
	vec[1] = vecB;
	vec[2] = vecC;
	vec[3] = vecD;
	
	res = (int**)malloc(sizeof(int*)*PATTERNS);
	for (int i = 0; i < PATTERNS; i++) {
		res[i] = (int*)malloc(sizeof(int)*lcnt);
		for (int j = 0; j < lcnt; j++) {
			res[i][j] = 0;
		}
	}
	DPRINT("All offsets: %d \n",test.max_offset);
	for (int i = 0; i < test.max_offset; i++){
		DPRINT("%d\t", test.offsets[i]);
	}
	DPRINT("\n");
	DPRINT("All Nodes offset values: \n");
	for (int i = 0; i <= ncnt; i++) {
		DPRINT("%d: %d\n", i,test.graph[i].offset);
	}


	dres = gpuLoadVectors(res, lcnt, PATTERNS);
	int* dvec = gpuLoadVectors(vec, 5, 4);
	ARRAY2D<int> inputArray = ARRAY2D<int>(dvec, 4, 5);
	dgraph = gpuLoadCircuit(test.graph,ncnt);
	dlines = gpuLoadLines(lgraph,lcnt);
	fans = gpuLoadFans(test.offsets,test.max_offset);
	loadLookupTables();
	runGpuSimulation(ARRAY2D<int>(dres,PATTERNS,lcnt), inputArray, test.graph,ARRAY2D<GPUNODE>(dgraph,1,ncnt),ARRAY2D<LINE>(dlines,PATTERNS,lcnt),fans, 1);
	DPRINT ("Max Node ID: %d\tLines: %d\n",ncnt,lcnt);
	PrintCircuit(graph,ncnt);
//	PrintLines(lgraph,lcnt);

	return 0;
}
