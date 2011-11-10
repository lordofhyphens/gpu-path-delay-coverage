#include "project.h"
#include <stdio.h>
#include <stdbool.h>

int procgate(NODE gnode, int settle); 
int main(int argc, char ** argv) {
	FILE *fisc;
	NODE graph[Mnod];
	fisc=fopen(argv[1],"r");
	ReadIsc(fisc,graph);
	int vecA[5] = {0,1,0,0,0};
	int vecB[5] = {1,0,0,1,1};
	int j = 0;

	// simulate logic values with vector A
	for (int i = 0; i < Mnod; i++) {
		if (graph[i].typ == INPT)
			graph[i].val = vecA[j];
		else
			graph[i].val = procgate(graph[i], true);
	}

	// simulate with vector B, determine transitions propagated.
	for (int i = 0; i < Mnod; i++) {
		if (graph[i].typ == INPT) {
			graph[i].val = vecB[j];
		}
		else {
			graph[i].val = procgate(graph[i], true);
		}
	}

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
