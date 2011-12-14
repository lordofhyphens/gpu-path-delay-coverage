
#include <stdio.h>
#include <stdbool.h>
#include "defines.h"
#include "simkernel.h"
#include "markkernel.h"
#include "coverkernel.h"
#include "iscas.h"
#include "gpuiscas.h"
#include "sort.h"

int main(int argc, char ** argv) {
	FILE *fisc, *fvec;
	int *dres, *fans, *vec;
	GPUNODE *dgraph;
	LINE *dlines;
	int** res;
	NODE* graph;
	graph = (NODE*)malloc(sizeof(NODE)*Mnod);
	LINE lgraph[Mnod];
	GPUNODE_INFO test;
	int vcnt, lcnt, ncnt, pis = 0; // count of lines in the circuit

	// Load circuit from file
	fisc=fopen(argv[1],"r");
	fvec=fopen(argv[2],"r");
	vcnt = readVectors(&vec, fvec);
	DPRINT("Vector count: %d, %d\n", vcnt, vcnt/5);

	ncnt = ReadIsc(fisc,graph);
	ncnt = topologicalSort(graph, ncnt);
	
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
		if (graph[i].typ == INPT)
			pis++;
	}

	res = (int**)malloc(sizeof(int*)*vcnt);
	for (int i = 0; i < vcnt; i++) {
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

	dres = gpuLoadVectors(res, lcnt, vcnt);
	int* dvec = gpuLoad1DVector(vec, pis, vcnt / pis);
	int *mergeresult;
	ARRAY2D<int> inputArray = ARRAY2D<int>(dvec, 4, 5);
	dgraph = gpuLoadCircuit(test.graph,ncnt);
	dlines = gpuLoadLines(lgraph,lcnt);
	fans = gpuLoadFans(test.offsets,test.max_offset);
	ARRAY2D<int> resArray = ARRAY2D<int>(dres,vcnt/pis,lcnt);
	ARRAY2D<GPUNODE> graphArray = ARRAY2D<GPUNODE>(dgraph,1,ncnt);
	float alltime, pass1, pass2, mark, merge,cover;
	pass1 = gpuRunSimulation(resArray, inputArray, test.graph,graphArray,fans, 1);
	TPRINT("Simulation Pass 1 time %fms\n", pass1);
//	debugSimulationOutput(resArray,1);
	gpuShiftVectors(dvec, pis, vcnt/pis);
	pass2 = gpuRunSimulation(resArray, inputArray, test.graph,graphArray,fans, 2);
	TPRINT("Simulation Pass 2 time %fms\n", pass2);
//	debugSimulationOutput(resArray,2);
	mark = gpuMarkPaths(resArray, test.graph, graphArray, fans);
	TPRINT("Path Mark time %fms\n",mark);
	merge = gpuMergeHistory(resArray, &mergeresult, test.graph, graphArray, fans);
	TPRINT("Path Merge time %fms\n",merge);

	debugMarkOutput(resArray);
	debugUnionOutput(ARRAY2D<int>(mergeresult,resArray.height, resArray.width));
	cover = gpuCountPaths(resArray,ARRAY2D<int>(mergeresult,resArray.height, resArray.width),test.graph,graphArray,fans);
	TPRINT("Path Coverage time %fms\n",cover);
	alltime = pass1 + pass2 + mark + merge + cover;
	debugCoverOutput(resArray);
	TPRINT("Total Path Count for vectors: %d\n", returnPathCount(resArray));
	TPRINT("Total time : %fms\n", alltime);
//	PrintCircuit(graph,ncnt);
//	PrintLines(lgraph,lcnt);

	return 0;
}
