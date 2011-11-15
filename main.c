#include "iscas.h"
#include <stdio.h>
#include <stdbool.h>

int procgate(NODE gnode, int settle); 	2	2	0		(10,3) 
2	2gat	1	0	0	1	2	2	0		(16,9) 
3	3gat	1	0	0	2	2	2	0		(8,1) (9,2) 
6	6gat	1	0	0	1	2	2	0		(11,6) 
7	7gat	1	0	0	1	2	2	0		(19,14) 
8	8fan	10	0	1	1	2	2	0	(3,-1) 	(10,4) 
9	9fan	10	0	1	1	2	2	0	(3,-1) 	(11,5) 
10	10gat	3	0	2	1	2	2	0	(1,-1) (8,-1) 	(22,15) 
11	11gat	3	0	2	2	2	2	0	(9,-1) (6,-1) 	(14,7) (15,8) 
14	14fan	10	0	1	1	2	2	0	(11,-1) 	(16,10) 
15	15fan	10	0	1	1	2	2	0	(11,-1) 	(19,13) 
16	16gat	3	0	2	2	2	2	0	(2,-1) (14,-1) 	(20,11) (21,12) 
19	19gat	3	0	2	1	2	2	0	(15,-1) (7,-1) 	(23,17) 
20	20fan	10	0	1	1	2	2	0	(16,-1) 	(22,16) 
21	21fan	10	0	1	1	2	2	0	(16,-1) 	(23,18) 

int main(int argc, char ** argv) {
	FILE *fisc;
	NODE graph[Mnod];
	fisc=fopen(argv[1],"r");
	ReadIsc(fisc,graph);
//	PrintCircuit(graph,50);
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
