
#include <stdio.h>
#include <stdbool.h>

#include "kernel.h"
#include "iscas.h"
#include "gpuiscas.h"

#include "defines.h"
int procgate(NODE gnode, int settle); 
int main(int argc, char ** argv) {
	FILE *fisc;
	int *dres, *fans;
	GPUNODE *dgraph;
	LINE *dlines;
	int** res;
	NODE graph[Mnod];
	LINE lgraph[Mnod];
	GPUNODE_INFO test;
	int lcnt, ncnt; // count of lines in the circuit

	// Load circuit from file
	fisc=fopen(argv[1],"r");
	ncnt = ReadIsc(fisc,graph);
	for (int i = 0; i < Mnod; i++)
		InitializeLines(lgraph, i);
	lcnt = EnumerateLines(graph,lgraph,ncnt);

	test = GraphsetToArrays(graph, lgraph, ncnt);
/*
	DPRINT("I\tLineID\tPrev\tNext\n");
	for(int i = 0; i < test.max_offset; i++) {
		DPRINT(" %d:\t%d\t%d\t%d\n",i,test.offsets[i],lgraph[test.offsets[i]].prev,lgraph[test.offsets[i]].next );
	}
*/
	DPRINT("ID:\tType\n");
	for(int i = 0; i < ncnt; i++) {
		DPRINT(" %d:\t%d\n",i,test.graph[i].type);
	}
	int vecA[5] = {1,0,0,1,1}; //v0
	int vecB[5] = {0,1,0,0,0}; //v1
	int vecC[5] = {1,0,1,0,1}; //v2
	int vecD[5] = {0,0,1,1,1}; //v3
	
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
	int j = 0;
	for (int i = 0; i < ncnt; i++) {
		if (test.graph[i].type == INPT) {
			res[0][test.offsets[test.graph[i].offset+test.graph[i].nfi]] = vecA[j];
			res[1][test.offsets[test.graph[i].offset+test.graph[i].nfi]] = vecB[j];
			res[2][test.offsets[test.graph[i].offset+test.graph[i].nfi]] = vecC[j];
			res[3][test.offsets[test.graph[i].offset+test.graph[i].nfi]] = vecD[j];
			j++;
		}
	}
/*
	DPRINT("Initial: \n");
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < lcnt; j++) {
			DPRINT("%d:%d\t%d\n", i, j, res[i][j]);
		}
	}
*/
	dres = gpuLoadVectors(res, lcnt, PATTERNS);
	dgraph = gpuLoadCircuit(test.graph,ncnt);
	dlines = gpuLoadLines(lgraph,lcnt);
	fans = gpuLoadFans(test.offsets,test.max_offset);
	loadLookupTables();
	runGpuSimulation(dres,lcnt,dgraph,test.graph,ncnt,dlines,lcnt,fans);
	DPRINT ("Max Node ID: %d\tLines: %d\n",ncnt,lcnt);
	PrintCircuit(graph,ncnt);
//	PrintLines(lgraph,lcnt);

	return 0;
}
