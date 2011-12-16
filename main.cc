
#include <stdio.h>
#include <stdbool.h>
#include "defines.h"
#include "simkernel.h"
#include "markkernel.h"
#include "coverkernel.h"
#include "iscas.h"
#include "serial.h"
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

	ncnt = ReadIsc(fisc,graph);
	ncnt = topologicalSort(graph, ncnt);
//	DPRINT("Initializing line structure.\n");	
	for (int i = 0; i < ncnt; i++)
		InitializeLines(lgraph, i);
//	DPRINT("Enumerating lines.\n");
	lcnt = EnumerateLines(graph,lgraph,ncnt);

	test = GraphsetToArrays(graph, lgraph, ncnt);

	for(int i = 0; i < ncnt; i++) {
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

//	PrintCircuit(graph,ncnt);
#ifndef CPUCOMPILE
// GPU implementation
	float alltime, pass1, pass2, mark, merge,cover;
	int* dvec = gpuLoad1DVector(vec, pis, vcnt / pis);
	int *mergeresult;


	DPRINT("Begin GPU Calculations\n");
	DPRINT("Load data into GPU Memory....");
	float elapsed = 0.0;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	dgraph = gpuLoadCircuit(test.graph,ncnt);
	fans = gpuLoadFans(test.offsets,test.max_offset);
	dres = gpuLoadVectors(res, lcnt, vcnt);

	ARRAY2D<int> inputArray = ARRAY2D<int>(dvec, vcnt/pis, pis);
	ARRAY2D<int> resArray = ARRAY2D<int>(dres,vcnt/pis,lcnt);
	ARRAY2D<GPUNODE> graphArray = ARRAY2D<GPUNODE>(dgraph,1,ncnt);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = (((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000.0) +0.5);
	DPRINT("...complete. Took %f ms\n", elapsed);

	pass1 = gpuRunSimulation(resArray, inputArray, test.graph,graphArray,fans, 1);
	
	TPRINT("Simulation Pass 1 time (GPU) %fms\n", pass1);
//	debugSimulationOutput(resArray,1);
	gpuShiftVectors(dvec, pis, vcnt/pis);
	pass2 = gpuRunSimulation(resArray, inputArray, test.graph,graphArray,fans, 2);

	TPRINT("Simulation Pass 2 time (GPU): %fms\n", pass2);
//	debugSimulationOutput(resArray,2);
	mark = gpuMarkPaths(resArray, test.graph, graphArray, fans);
	TPRINT("Path Mark time (GPU): %fms\n",mark);
	debugMarkOutput(resArray);
	merge = gpuMergeHistory(resArray, &mergeresult, test.graph, graphArray, fans);
//	debugMarkOutput(ARRAY2D<int>(mergeresult,resArray.height, resArray.width));
	TPRINT("Path Merge time (GPU): %fms\n",merge);
	cover = gpuCountPaths(resArray,ARRAY2D<int>(mergeresult,resArray.height, resArray.width),test.graph,graphArray,fans);
	TPRINT("Path Coverage time (GPU): %fms\n",cover);
	alltime = pass1 + pass2 + mark + merge + cover;

	TPRINT("Total Path Count for vectors (GPU): %d\n", returnPathCount(resArray));
	TPRINT("Total time (GPU) : %fms\n", alltime);
#endif
// Serial implementation
	float alltime_S, pass1_s, pass2_s, mark_s, merge_s, cover_s;
	int *cres, *cfans, *cvec; // serial implementation
	GPUNODE *cgraph;
	int *mergeserial;
	
	cres = cpuLoadVectors(res, lcnt, vcnt);
	cvec = cpuLoad1DVector(vec, pis, vcnt / pis);
	cfans = cpuLoadFans(test.offsets,test.max_offset);
	cgraph = cpuLoadCircuit(test.graph,ncnt);

	ARRAY2D<int> sResArray = ARRAY2D<int>(cres,vcnt/pis,lcnt);
	ARRAY2D<GPUNODE> sGraphArray = ARRAY2D<GPUNODE>(cgraph,1,ncnt);
	ARRAY2D<int> sInputArray = ARRAY2D<int>(cvec, 4, 5);

	pass1_s = cpuRunSimulation(sResArray, sInputArray, test.graph,sGraphArray,cfans, 1);
	TPRINT("Simulation Pass 1 time (serial) %fms\n", pass1_s);
	cpuShiftVectors(cvec, pis, vcnt/pis);
	pass2_s = cpuRunSimulation(sResArray, sInputArray, test.graph,sGraphArray,cfans, 2);
	TPRINT("Simulation Pass 2 time (serial) %fms\n", pass2_s);
	mark_s = cpuMarkPaths(sResArray, test.graph, sGraphArray, cfans);
	TPRINT("Path Mark time (serial) %fms\n",mark_s);
//	debugCpuMark(sResArray);
	merge_s = cpuMergeHistory(sResArray, &mergeserial, test.graph, sGraphArray, cfans);
	TPRINT("Path Merge time %fms\n",merge);

	cover = cpuCountPaths(sResArray,ARRAY2D<int>(mergeserial,sResArray.height, sResArray.width),test.graph,sGraphArray,fans);
	TPRINT("Path Coverage time (serial) %fms\n",cover_s);
	alltime = pass1_s + pass2_s + mark_s + merge_s + cover_s;

	TPRINT("Total Path Count for vectors (serial): %d\n", sReturnPathCount(sResArray));
	TPRINT("Total time (serial) : %fms\n", alltime);

	return 0;
}
