#include "serial.h"
#include "iscas.h"
#include "gpuiscas.h"
#include <ctime>
#include <time.h>
#include <unistd.h>

void cpuMerge(int h, int w, int* input, int* results, int width) {
	int merge[2][2] = {{0,1},{1,1}};
	int *r,result, i;
	if (w < width) {
		result = 0;
		for (i = 0; i <= h; i++) {
			r = (int*)(input + i*width);
			DPRINT("r[w] = %d, result = %d\n",r[w],result);
			assert(r[w] < 2 && result < 2 && r[w] >= 0 && result >= 0);
			result = merge[result][r[w]];
		}
		r = (int*)((char*)results + sizeof(int)*width*h);
		r[w] = result;
	}
}
void cpuMarkPathSegments(int *results, int tid, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
	int and2_output_prop[2][4][4]= {{{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}} ,{{0,0,1,0},{0,0,1,1},{1,1,1,1},{0,1,1,1}}};
	int and2_input_prop[2][4][4] = {{{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}} ,{{0,0,0,0},{0,0,1,1},{0,0,1,0},{0,0,1,1}}};
	int or2_output_prop[2][4][4] = {{{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}} ,{{0,0,1,1},{0,0,0,0},{1,0,1,1},{1,0,1,1}}};
	int or2_input_prop[2][4][4]  = {{{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}} ,{{0,0,1,1},{0,0,0,0},{0,0,1,1},{0,0,0,1}}};
	int from_prop[16]      =  {0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1};
	int inpt_prop[2][4] = {{0,0,0,0},{0,0,1,1}};
	int nfi, goffset,val;
	int *rowResults, *row;
	if (tid < height) {
		DPRINT("TID: %d\n", tid);
		row = (int*)((char*)results + tid*(width)*sizeof(int));
		rowResults = (int*)malloc(sizeof(int)*width);
		for (int i = 0; i < width; i++) {
			rowResults[i] = UNINITIALIZED;
		}
		for (int i = ncount; i >= 0; i--) {
			val = UNINITIALIZED;
			goffset = node[i].offset;
			nfi = node[i].nfi;
			// switching based on value causes divergence, switch based on node type.
			switch(node[i].type) {
				
				case FROM:
					// For FROM, only set the "input" line if it hasn't already
					// been set (otherwise it'll overwrite the decision of
					// another system somewhere else.
					if (rowResults[fans[goffset]] == UNINITIALIZED) {
						val = inpt_prop[row[fans[goffset]]][rowResults[fans[goffset+nfi]]];
						rowResults[fans[goffset]] = val;
						rowResults[fans[goffset+nfi]] = val;
					} else {
						val = inpt_prop[row[fans[goffset]]][rowResults[fans[goffset+nfi]]];
						rowResults[fans[goffset+nfi]] = val;
					}
					break;
					// For the standard gates, setting three values -- both the input lines and the output line.
				case NAND:
				case AND:
					rowResults[fans[goffset]] = and2_input_prop[row[fans[goffset+nfi]]-1][row[fans[goffset]]][row[fans[goffset+1]]];
					rowResults[fans[goffset+1]] = and2_input_prop[row[fans[goffset+nfi]]-1][row[fans[goffset+1]]][row[fans[goffset]]];
					rowResults[fans[goffset+nfi]] = and2_output_prop[row[fans[goffset+nfi]]-1][row[fans[goffset]]][row[fans[goffset+1]]];
					break;
				case OR:
				case NOR:
					rowResults[fans[goffset]] = or2_input_prop[row[fans[goffset+nfi]]-1][row[fans[goffset]]][row[fans[goffset+1]]];
					rowResults[fans[goffset+1]] = or2_input_prop [row[fans[goffset+nfi]]-1][row[fans[goffset+1]]][row[fans[goffset]]];
					rowResults[fans[goffset+nfi]] = or2_output_prop[row[fans[goffset+nfi]]-1][row[fans[goffset]]][row[fans[goffset+1]]];
					break;
				case XOR:
				case XNOR:
				default:
					// if there is a transition that will propagate, set = to some positive #?
					break;
			}
		}
		for (int i = 0; i < width; i++) {
			row[i] = rowResults[i] * (tid+1);
		}
		free(rowResults);
	}
}

void cpuCountCoverage(int toffset, int tid, int *results, int *history, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
	int nfi, goffset, vecnum = toffset + tid, val, tempr, temph;
	int *row, *rowHistory;
	int *rowResults,*historyResults;
	int h_cache[2048];
	int r_cache[2048];
	// For every node, count paths through this node
	if (tid < height) {
		rowResults = (int*)malloc(sizeof(int)*width);
		historyResults = (int*)malloc(sizeof(int)*width);
		row = (int*)((char*)results + vecnum*(width)*sizeof(int));
		rowHistory = (int*)((char*)history + (vecnum-1)*(width)*sizeof(int));
		for (int i = 0; i < width; i++) {
			rowResults[i] = 0; 
			historyResults[i] = 0;
		}
		if (vecnum == 0) {
			// handle the initial vector separately, just count paths.
			for (int c = ncount; c >= 0; c--) {
				nfi = node[c].nfi;
				goffset = node[c].offset;
				if (node[c].po) {
					rowResults[fans[goffset+nfi]] = row[fans[goffset+nfi]]*1;
				}
				switch (node[c].type) {
					case 0: break;
					case INPT: break;
					case FROM:
						rowResults[fans[goffset]] += row[fans[goffset+nfi]]*(rowResults[fans[goffset]]);
					break;
					default:
						for (int i = 0; i < nfi;i++)
							rowResults[fans[goffset+i]] = rowResults[fans[goffset+nfi]];

				}
			}
			for (int i = 0; i < width; i++) {
				row[i] = rowResults[i];
			}

		} else {
			for (int c = ncount; c >= 0; c--) {
				nfi = node[c].nfi;
				goffset = node[c].offset;
				if (node[c].po) {
					tempr = 0;
					temph = rowHistory[fans[goffset+nfi]]*1;
					val = row[fans[goffset+nfi]] > rowHistory[fans[goffset+nfi]];
					r_cache[0] = tempr;
					r_cache[1] = tempr + temph;
					h_cache[0] = temph;
					h_cache[1] = 0;
					rowResults[fans[goffset+nfi]] = r_cache[val];
					historyResults[fans[goffset+nfi]] = h_cache[val];
				}
				switch (node[c].type) {
					case 0: continue;
					case INPT: break;
					case FROM:
							   tempr = rowResults[fans[goffset]];
							   tempr += row[fans[goffset+nfi]]*(rowResults[fans[goffset+nfi]]);
							   temph = historyResults[fans[goffset]];
							   temph += rowHistory[fans[goffset+nfi]]*(historyResults[fans[goffset+nfi]]);
							   val = row[fans[goffset]] > rowHistory[fans[goffset]];
							   r_cache[0] = tempr;
							   r_cache[1] = tempr + temph;
							   h_cache[0] = temph;
							   h_cache[1] = 0;
							   rowResults[fans[goffset]] = r_cache[val];
							   historyResults[fans[goffset]] = h_cache[val];
							   break;
					default:
							   for (int i = 0; i < nfi;i++) {
								   rowResults[fans[goffset+i]] = rowResults[fans[goffset+nfi]];
								   historyResults[fans[goffset+i]] = historyResults[fans[goffset+nfi]];
								   val = row[fans[goffset+i]] > rowHistory[fans[goffset+i]];
								   r_cache[0]     = rowResults[fans[goffset+i]];
								   r_cache[1] = rowResults[fans[goffset+i]] + historyResults[fans[goffset+i]];
								   h_cache[0]     = historyResults[fans[goffset+i]];
								   h_cache[1] = 0;
								   rowResults[fans[goffset+i]] = r_cache[val];
								   historyResults[fans[goffset+i]] = h_cache[val];
							   }
				} 
			}
			for (int i = 0; i < width; i++) {
				row[i] = rowResults[i];
			}
		}

		free(rowResults);
		free(historyResults);
	}
}


void sINPT_gate(int i, int j, int pi, int* row, int height, ARRAY2D<int> input, GPUNODE* graph, int* fans,int pass) {
	int tid, val, nfi = graph[i].nfi, goffset = graph[i].offset;
	int stable[2][2] = {{S0, T0}, {T1, S1}};
	int from[4] = {0, 0, 1, 1};
	val = *(input.data+(pi+input.width*j));
//	DPRINT(" val: %d\n", val);
	if (pass > 1) {
		assert(row[fans[goffset+nfi]] < 4);
	} else {
		assert(row[fans[goffset+nfi]] < 2);
	}
	assert(val < 2);
	if (pass > 1) {
		row[fans[goffset+nfi]] = stable[from[row[fans[goffset+nfi]]]][val];  
	} else {
		row[fans[goffset+nfi]] = val;
	}
}

void sLOGIC_gate(int i, int tid, GPUNODE* node, int* fans, int* results, size_t height, size_t width , int pass) {
	int nand2[4][4] = {{1, 1, 1, 1}, {1, 0, 1, 0}, {1, 1, 1, 1}, {1, 0, 1, 0}};
	int and2[4][4]  = {{0, 0, 0, 0}, {0, 1, 0, 1}, {0, 0, 0, 0}, {0, 1, 0, 1}};
	int nor2[4][4]  = {{1, 0, 1, 0}, {0, 0, 0, 0}, {1, 0, 1, 0}, {0, 0, 0, 0}};
	int or2[4][4]   = {{0, 1, 0, 1}, {1, 1, 1, 1}, {0, 1, 0, 1}, {1, 1, 1, 1}};
	int xnor2[4][4] = {{1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}};
	int xor2[4][4]  = {{0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}};
	int stable[2][2] = {{S0, T0}, {T1, S1}};
	int from[4] = {0, 0, 1, 1};
	int j = 1;
	int goffset,nfi;
	int val=0;
	int *row = (int*)results + tid*(width); // get the current row?
	goffset = node[i].offset;
	nfi = node[i].nfi;
	if (pass > 1) {
		val = from[row[fans[goffset]]];
	} else{ 
		val = row[fans[goffset]];
	}
	while (j < nfi) {
		switch(node[i].type) {
			case XOR:
				val = xor2[val][row[fans[goffset+j]]]; break;
			case XNOR:
				val = xnor2[val][row[fans[goffset+j]]]; break;
			case OR:
				val = or2[val][row[fans[goffset+j]]]; break;
			case NOR:
				val = nor2[val][row[fans[goffset+j]]]; break;
			case AND:
				val = and2[val][row[fans[goffset+j]]]; break;
			case NAND:
				val = nand2[val][row[fans[goffset+j]]]; break;
			default:
				val = from[row[fans[goffset+j]]];
		}
		j++;
	}

	assert(row[fans[goffset+nfi]] < 4);
	assert(val < 2);
	if (pass > 1 && node[i].type != FROM) {
		row[fans[goffset+nfi]] = stable[row[fans[goffset+nfi]]][val];  
	} else {
		row[fans[goffset+nfi]] = val;
	}
}

float cpuRunSimulation(ARRAY2D<int> results, ARRAY2D<int> inputs, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan, int pass) {
	int piNumber = 0, curPI = 0;
	int *row;
	float elapsed = 0.0;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int j = 0; j < results.height; j++) {
		for (int i = 0; i < dgraph.width; i++) {
			curPI = piNumber;
			switch (graph[i].type) {
				case 0:
					continue;
				case INPT:
//					DPRINT("INPT Gate");
					sINPT_gate(i, j, curPI, results.data, results.height, inputs, dgraph.data, fan, pass);
					piNumber++;
					break;
				default:
//					DPRINT("Logic Gate, %d - %d type %d\n", j, i, graph[i].type);
					sLOGIC_gate(i, j, dgraph.data, fan, results.data, results.height, results.width, pass);
					break;
			}
//			DPRINT("\n");
		}
		piNumber = 0;
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = (((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000.0) +0.5);
	return elapsed;
}

int* cpuLoad1DVector(int* input, size_t width, size_t height) {
	int *tgt;
	tgt = (int*)malloc(sizeof(int)*(width+1)*(height+1));
	memcpy(tgt, input,sizeof(int)*(width+1)*(height+1));
	return tgt;
}

int* cpuLoadVectors(int** input, size_t width, size_t height) {
	int *tgt;
	tgt = (int*)malloc(sizeof(int)*(width+1)*(height+1));
	int *row;
	for (int i = 0; i < height; i++) {
		row = (int*)((char*)tgt + i*(width)*sizeof(int));
		memcpy(row, input[i],sizeof(int)*(width+1));
#ifndef NDEBUG
		int *tmp = (int*)malloc(sizeof(int)*width);
		for (int r =0; r <= width; r++)
			tmp[r] = -1;
		memcpy(tmp, row, sizeof(int)*(width+1));
		for (int j = 0; j <= width; j++) {
			assert(input[i][j]==tmp[j]);
		}
		free(tmp);
#endif // debugging memory check and assertion
	}
	return tgt;
}

GPUNODE* cpuLoadCircuit(const GPUNODE* graph, int maxid) {
	GPUNODE *devAr, *testAr;
	devAr = (GPUNODE*)malloc(sizeof(GPUNODE)*(1+maxid));
	memcpy(devAr, graph, (maxid+1) * sizeof(GPUNODE));
	
	return devAr;
}
LINE* cpuLoadLines(LINE* graph, int maxid) {
	LINE *devAr;
	devAr = (LINE*)malloc(sizeof(LINE)*maxid);
	memcpy(devAr, graph, sizeof(LINE)*maxid);
	return devAr;
}
int* cpuLoadFans(int* offset, int maxid) {
	int* devAr;
	devAr = (int*)malloc(sizeof(int)*maxid);
	memcpy(devAr, offset, sizeof(int)*maxid);
	return devAr;
}
void cpuShiftVectors(int* input, size_t width, size_t height) {
	int* tgt, *z;
	// create a temporary buffer area on the device
	tgt = (int*)malloc(sizeof(int)*(width));
	memcpy(tgt, input,sizeof(int)*(width));
	memcpy(input, input+width,sizeof(int)*(width)*(height-1));
	memcpy(input+(height-1)*(width),tgt, sizeof(int)*(width));
	for (int i = 0; i < height*width; i++) {
		assert(input[i] < 4);
	}
	free(tgt);
}
float cpuMarkPaths(ARRAY2D<int> results, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph,  int* fan) {
	float elapsed = 0.0;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < results.height; i++) {
		DPRINT("Running TID: %d ", i);
		cpuMarkPathSegments(results.data, i, dgraph.data, fan, results.width, results.height, dgraph.width);
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = (((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000.0) +0.5);
	return elapsed;
}
float cpuMergeHistory(ARRAY2D<int> input, int** mergeresult, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan) {
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	*(mergeresult) = (int*)malloc(sizeof(int)*input.height*input.width);
	for (int i = 0; i < input.height; i++)
		for (int j = 0; j< input.width; j++) {
			DPRINT("Beginning merge %d,%d\n", i, j);
			cpuMerge(i,j,input.data, *mergeresult, input.width);
			DPRINT("Finished merge %d,%d\n", i, j);
			memcpy(*mergeresult, input.data, input.bwidth());
		}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = (((stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000.0) +0.5);
	return elapsed;
}


void cpuSumAll(int toffset, int tid, int *results, int *history, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
	int nfi, goffset;
	int *row;
	int sum;
	if (tid < 1) {
		sum = 0;
		for (int j = 0; j < height; j++) {
			row = (int*)((char*)results + j*(width)*sizeof(int));
			for (int c = ncount; c >= 0; c--) {
				goffset = node[c].offset;
				nfi = node[c].nfi;
				if (node[c].type == INPT)
					sum = sum + row[fans[goffset+nfi]];
			}
		}
		row = (int*)((char*)results);
		row[0] = sum;
	}
}

float cpuCountPaths(ARRAY2D<int> results, ARRAY2D<int> history, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan) {
	float elapsed = 0.0;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	DPRINT("rw %d, rh %d, gw %d, gh %d\n", results.width, results.height, dgraph.width, dgraph.height);
	for (int j = 0; j < results.height; j++) {
		cpuCountCoverage(0, j,results.data, history.data,dgraph.data, fan, results.width, results.height, dgraph.width);
	}
	cpuSumAll(0, 0, results.data, history.data,dgraph.data, fan, results.width, results.height, dgraph.width);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = (stop.tv_nsec - start.tv_nsec) / 1000000.0;
	return elapsed;
}

int sReturnPathCount(ARRAY2D<int> results) {
	int tmp;
	memcpy(&tmp, results.data, sizeof(int));
	return tmp;
}

void debugCpuMark(ARRAY2D<int> results) {
#ifndef NDEBUG
	int *row;
	DPRINT("Post-mark results\n");
	DPRINT("Line:   \t");
	for (int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (int r = 0;r < results.height; r++) {
		row = (int*)((char*)results.data + r*results.bwidth()); // get the current row?
		DPRINT("%s %d:\t","Vector",r);
		for (int i = 0; i < results.width; i++) {
			DPRINT("%2d ", row[i]);
		}
		DPRINT("\n");
	}
#endif
}
