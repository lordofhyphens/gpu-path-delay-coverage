#include <stdio.h>
#include <stdbool.h>
#include "defines.h"
#include "iscas.h"
#include "gpuiscas.h"
#include "simkernel.h"
#include "markkernel.h"
#include "coverkernel.h"
#include "serial.h"
#include "sort.h"
#include "array2d.h"

int main(int argc, char ** argv) {
	FILE *fisc, *fvec;
	int *fans, *vec,*dvec, levels;
	GPUNODE *dgraph;
	NODE* graph;
	graph = (NODE*)malloc(sizeof(NODE)*Mnod);
	LINE lgraph[Mnod];
	GPUNODE_INFO test;
	int vcnt, lcnt, ncnt, pis = 0; // count of lines in the circuit

	// Load circuit from file
	fisc=fopen(argv[1],"r");
	fvec=fopen(argv[2],"r");
	vcnt = readVectors(&vec, fvec);
	DPRINT("%d characters read.\n",vcnt);
//	vec = loadPinned(vec, vcnt);

	ncnt = ReadIsc(fisc,graph);
	DPRINT("Sorting Circuit\n");
	ncnt = topologicalSort(graph, ncnt);
	DPRINT("Levelizing circuit...");
	levels = levelize(graph,ncnt);
	levelSort(graph,ncnt);
	DPRINT("...complete. Maximum level = %d\n", levels);
//	DPRINT("Initializing line structure.\n");	
	for (int i = 0; i < ncnt; i++)
		InitializeLines(lgraph, i);
	DPRINT("Enumerating lines....");
	lcnt = EnumerateLines(graph,lgraph,ncnt);
	DPRINT(" %d lines complete.\n", lcnt);
	DPRINT("Copying to flat arrays...");
	test = GraphsetToArrays(graph, lgraph, ncnt);
	DPRINT("complete.\n");

	for(int i = 0; i < ncnt; i++) {
		if (graph[i].typ == INPT)
			pis++;
	}
	DPRINT("%d primary inputs, %d input vectors.\n", pis, vcnt);
//	PrintCircuit(graph,ncnt);
for (int i = 0; i < test.max_offset; i++) {
	if (test.offsets[i] < 0)
		printf("%d ", test.offsets[i]);
}
#ifndef CPUCOMPILE
// GPU implementation
	float alltime, pass1=0.0, pass2=0.0, mark = 0.0, merge = 0.0,cover = 0.0;
	ARRAY2D<char> mergeresult;

//	DPRINT("Begin GPU Calculations\n");
//	DPRINT("Load data into GPU Memory....");
	float elapsed = 0.0;
	timespec start, stop;
//	DPRINT("Getting time.");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

//	DPRINT("Loading circuit...");
	dgraph = gpuLoadCircuit(test.graph,ncnt);
//	DPRINT("complete.\n");
	fans = gpuLoadFans(test.offsets,test.max_offset);
//	DPRINT("Allocating GPU results memory....");
//	DPRINT("...complete.\n");
//	DPRINT("W: %d H: %d\n", pis, (vcnt/pis));
//	DPRINT("Loading test vectors into GPU Memory...");
	dvec = gpuLoad1DVector(vec, pis, vcnt);
	freeMemory(vec);
//	DPRINT("...complete.\n");

	ARRAY2D<int> inputArray = ARRAY2D<int>(dvec, vcnt, pis);
	ARRAY2D<char> resArray = gpuAllocateResults(lcnt, vcnt);
	ARRAY2D<GPUNODE> graphArray = ARRAY2D<GPUNODE>(dgraph,1,ncnt);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
//	DPRINT("GPU Memory Time: %f ms\n", elapsed);

	pass1 = gpuRunSimulation(resArray, inputArray, test.graph,graphArray,fans, levels, 1);
	
	TPRINT("Simulation Pass 1 time (GPU): %f ms\n", pass1);
//	debugSimulationOutput(resArray,1);
	gpuShiftVectors(dvec, pis, vcnt);
	pass2 = gpuRunSimulation(resArray, inputArray, test.graph,graphArray,fans, levels, 2);
//	debugSimulationOutput(resArray,2);
	TPRINT("Simulation Pass 2 time (GPU): %f ms\n", pass2);
	freeMemory(inputArray.data);

	mergeresult = gpuAllocateResults(lcnt, vcnt);
	mark = gpuMarkPaths(resArray, mergeresult, test.graph, graphArray, fans);
	TPRINT("Path Mark time (GPU): %fms\n",mark);
	DPRINT("Fiddling with memory...");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	gpuArrayCopy(resArray, mergeresult);
	clearMemory(mergeresult);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed += floattime(diff(start, stop));
	DPRINT("...complete.\n");
//	debugMarkOutput(resArray);
	merge = gpuMergeHistory(resArray, &(mergeresult), test.graph, graphArray, fans);
//	debugMarkOutput(mergeresult);
	TPRINT("Path Merge time (GPU): %fms\n",merge);
	cover = gpuCountPaths(resArray,mergeresult,test.graph,graphArray,fans);
//	debugCoverOutput(resArray);
	TPRINT("Path Coverage time (GPU): %fms\n",cover);
	alltime = pass1 + pass2 + mark + merge + cover;

	TPRINT("Total Path Count for vectors (GPU): %d\n", returnPathCount(resArray));
#endif
// Serial implementation
	float alltime_s=0.0, pass1_s=0.0, pass2_s=0.0, mark_s=0.0, merge_s=0.0, cover_s=0.0;
	int *cres, *cfans, *cvec; // serial implementation
	GPUNODE *cgraph;
	cres = cpuAllocateResults(lcnt, vcnt);
	cvec = cpuLoad1DVector(vec, pis, vcnt);
	cfans = cpuLoadFans(test.offsets,test.max_offset);
	cgraph = cpuLoadCircuit(test.graph,ncnt);

	ARRAY2D<int> sResArray = ARRAY2D<int>(cres,vcnt,lcnt);
	ARRAY2D<GPUNODE> sGraphArray = ARRAY2D<GPUNODE>(cgraph,1,ncnt);
	ARRAY2D<int> sInputArray = ARRAY2D<int>(cvec, 4, 5);

	pass1_s = cpuRunSimulation(sResArray, sInputArray, test.graph,sGraphArray,cfans, 1);
	TPRINT("Simulation Pass 1 time (serial): %f ms\n", pass1_s);
//	debugCpuSimulationOutput(sResArray,1);
	cpuShiftVectors(cvec, pis, vcnt);
	pass2_s = cpuRunSimulation(sResArray, sInputArray, test.graph,sGraphArray,cfans, 2);
//	debugCpuSimulationOutput(sResArray,2);
	TPRINT("Simulation Pass 2 time (serial): %f ms\n", pass2_s);
	mark_s = cpuMarkPaths(sResArray, test.graph, sGraphArray, cfans);
	TPRINT("Path Mark time (serial) %fms\n",mark_s);
//	debugCpuMark(sResArray);
	int* pathcount_s = (int*)malloc(sizeof(int));
	cover_s = cpuCountPaths(sResArray,test.graph,sGraphArray,cfans, pathcount_s);
	TPRINT("Path Coverage time (serial) %fms\n",cover_s);
	alltime_s = pass1_s + pass2_s + mark_s + merge_s + cover_s;
#ifndef CPUCOMPILE
	freeMemory(mergeresult.data);
	freeMemory(resArray.data);
	freeMemory(inputArray.data);
#endif
	TPRINT("Total Path Count for vectors (serial): %d\n", *pathcount_s);
#ifndef CPUCOMPILE	
	TPRINT("%d, %f,%f,", vcnt,elapsed, alltime);
	TPRINT("%f\n", alltime_s);
#else
	TPRINT("Total CPU Time: %f\n", alltime_s);
#endif 
	return 0;
}
