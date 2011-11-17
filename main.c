#include "iscas.h"
#include <stdio.h>
#include <stdbool.h>

int procgate(NODE gnode, int settle); 

int main(int argc, char ** argv) {
	FILE *fisc;
	NODE graph[Mnod];
	LINE lgraph[Mnod];
	GPUNODE_INFO test;
	int lcnt, ncnt; // count of lines in the circuit
	fisc=fopen(argv[1],"r");
	ncnt = ReadIsc(fisc,graph);
	for (int i = 0; i < Mnod; i++)
		InitializeLines(lgraph, i);
	lcnt = EnumerateLines(graph,lgraph);

	test = GraphsetToArrays(graph, lgraph, ncnt);
	printf("I\tLineID\tPrev\tNext\n");
	for(int i = 0; i < test.max_offset; i++) {
		printf(" %d:\t%d\t%d\t%d\n",i,test.offsets[i],lgraph[test.offsets[i]].prev,lgraph[test.offsets[i]].next );
	}
	printf ("Max Node ID: %d\tLines: %d\n",ncnt,lcnt);
	int vecA[5] = {0,1,0,0,0};
	int vecB[5] = {1,0,0,1,1};
	int j = 0;

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
