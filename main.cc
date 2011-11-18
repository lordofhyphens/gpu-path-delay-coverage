#define PATTERNS 2

#include <stdio.h>
#include <stdbool.h>

#include "kernel.h"

#include "iscas.h"
#include "gpuiscas.h"

int procgate(NODE gnode, int settle); 
int main(int argc, char ** argv) {
	FILE *fisc;
	int *dres, *fans;
	NODE *dgraph;
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
	lcnt = EnumerateLines(graph,lgraph);

	test = GraphsetToArrays(graph, lgraph, ncnt);
/*
	printf("I\tLineID\tPrev\tNext\n");
	for(int i = 0; i < test.max_offset; i++) {
		printf(" %d:\t%d\t%d\t%d\n",i,test.offsets[i],lgraph[test.offsets[i]].prev,lgraph[test.offsets[i]].next );
	}
*/
	int vecA[5] = {0,1,0,0,0};
	int vecB[5] = {1,0,0,1,1};
	
	res = (int**)malloc(sizeof(int*)*PATTERNS);
	for (int i = 0; i < PATTERNS; i++) {
		res[i] = (int*)malloc(sizeof(int)*lcnt);
		for (int j = 0; j < lcnt; j++) {
			res[i][j] = 0;
		}
	}
	printf("All offsets: \n");
	for (int i = 0; i < test.max_offset; i++){
		printf("%d\t", test.offsets[i]);
	}
	printf("\n");
	printf("All Nodes offset values: \n");
	for (int i = 0; i < ncnt; i++) {
		printf("%d: %d\n", i,test.graph[i].offset);
	}
	int j = 0;
	for (int i = 0; i < ncnt; i++) {
		if (test.graph[i].type == INPT) {
			printf("line: %d, vecA, %d: %d\t vecB, %d: %d\n",test.offsets[test.graph[i].offset+test.graph[i].nfi],j,vecA[j],j,vecB[j]);
			printf("Setting res[0].\n");
			res[0][test.offsets[test.graph[i].offset+test.graph[i].nfi]] = vecA[j];
			printf("Setting res[1].\n");
			res[1][test.offsets[test.graph[i].offset+test.graph[i].nfi]] = vecB[j];
			j++;
		}
	}
	printf("Initial: \n");
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < lcnt; j++) {
			printf("%d:%d\t%d\n", i, j, res[i][j]);
		}
	}

	size_t pitch;
	dres = gpuLoadVectors(res, lcnt, PATTERNS);
	dgraph = gpuLoadCircuit(graph,ncnt);
	dlines = gpuLoadLines(lgraph,lcnt);
	fans = gpuLoadFans(test.offsets,test.max_offset);
	runGpuSimulation(dres,lcnt,test.graph,ncnt,dlines,lcnt,fans);
	printf ("Max Node ID: %d\tLines: %d\n",ncnt,lcnt);
	PrintCircuit(graph,ncnt);
	PrintLines(lgraph,lcnt);

	return 0;
}

int procgate(NODE gnode, int settle) {
	switch(gnode.val) {
		case AND:
			break;
		case NAND:
			break;
		case OR:
			break;
		case NOR:
			break;
		case XOR:
			break;
		case XNOR:
			break;
		case BUFF:
			break;
		case NOT:
			break;
		case FROM:
			break;
	}
	return 0;
}
