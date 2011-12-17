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
//			DPRINT("i = %d, w = %d, r[w] = %d, result = %d\n",i,w, r[w],result);
			assert(r[w] < 2 && result < 2 && r[w] >= 0 && result >= 0);
			result |= r[w];
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
	int rowids[1000];
	int *rowResults, *row;
	if (tid < height) {
		row = (int*)((char*)results + tid*(width)*sizeof(int));
		rowResults = (int*)malloc(sizeof(int)*width);
		for (int i = ncount; i >= 0; i--) {
			val = UNINITIALIZED;
			goffset = node[i].offset;
			nfi = node[i].nfi;
			for (int j =0; j < nfi; j++) {
				rowids[j] = fans[goffset+j];
			}
			// switching based on value causes divergence, switch based on node type.
			val = (row[i] >1);
			if (node[i].po) {
				rowResults[i] = val;
			}
			switch(node[i].type) {
				case FROM:
					// For FROM, only set the "input" line if it hasn't already
					// been set (otherwise it'll overwrite the decision of
					// another system somewhere else.
					val = inpt_prop[row[rowids[0]]][row[i]];
					rowResults[rowids[0]] |= val;
					rowResults[i] &= val;
					break;
					// For the standard gates, setting three values -- both the input lines and the output line.
				case NAND:
				case AND:
					rowResults[i] &= val && and2_output_prop[row[i]-1][row[rowids[0]]][row[rowids[1]]];
					rowResults[rowids[0]] = val&& and2_input_prop[row[i]-1][row[rowids[0]]][row[rowids[1]]];
					rowResults[rowids[1]] = val&& and2_input_prop[row[i]-1][row[rowids[1]]][row[rowids[0]]];
					break;
				case OR:
				case NOR:
					rowResults[rowids[0]] = val&&or2_input_prop[row[i]-1][row[rowids[0]]][row[rowids[1]]];
					rowResults[rowids[1]] = val&&or2_input_prop [row[i]-1][row[rowids[1]]][row[rowids[0]]];
					rowResults[i] &= val&&or2_output_prop[row[i]-1][row[rowids[0]]][row[rowids[1]]];
					break;
				case XOR:
				case XNOR:
				default:
					// if there is a transition that will propagate, set = to some positive #?
					break;
			}
		}
		for (int i = ncount; i >= 0; i--) {
			nfi = node[i].nfi;
			if (tid == 0) {
				goffset = node[i].offset;
				// preload all of the fanin line #s for this gate to shared memory.
				for (int j = 0; j < nfi;j++) 
					rowids[j] = fans[goffset+j];
			}
			int bin = rowResults[i];
			if (node[i].type == INPT) {
				continue;
			}
			for (int j = 0; j < node[i].nfi; j++) {
				rowResults[rowids[j]] &= bin;
			}

		}
		for (int i = 0; i < width; i++) {
			row[i] = rowResults[i];
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
					rowResults[c] = row[c]*1;
				}
				switch (node[c].type) {
					case 0: break;
					case INPT: break;
					case FROM:
						rowResults[fans[goffset]] += row[c]*(rowResults[fans[goffset]]);
					break;
					default:
						for (int i = 0; i < nfi;i++)
							rowResults[fans[goffset+i]] = rowResults[c];

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

void cpuSimulate(GPUNODE* graph, int* res, int* input, int* fans, size_t iwidth, size_t width, size_t height, int pass, int tid) {
	int nand2[4][4] = {{1, 1, 1, 1}, {1, 0, 1, 0}, {1, 1, 1, 1}, {1, 0, 1, 0}};
	int and2[4][4]  = {{0, 0, 0, 0}, {0, 1, 0, 1}, {0, 0, 0, 0}, {0, 1, 0, 1}};
	int nor2[4][4]  = {{1, 0, 1, 0}, {0, 0, 0, 0}, {1, 0, 1, 0}, {0, 0, 0, 0}};
	int or2[4][4]   = {{0, 1, 0, 1}, {1, 1, 1, 1}, {0, 1, 0, 1}, {1, 1, 1, 1}};
	int xnor2[4][4] = {{1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}};
	int xor2[4][4]  = {{0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}};
	int stable[2][2] = {{S0, T0}, {T1, S1}};
	int from[4] = {0, 0, 1, 1};
	int notl[4] = {1, 0, 1, 0};
	char rowids[1000]; // handle up to fanins of 1000 / 
	int piNumber = 0, pi = 0;
	int *row;
	int goffset, nfi, val, j,type, r;
	if (tid < height) {
		row = (res + tid*width); // get the current row?
		for (int i = 0; i <= width; i++) {
			nfi = graph[i].nfi;
			goffset = graph[i].offset;
			// preload all of the fanin line #s for this gate to shared memory.
			for (int N = 0; N < nfi;N++)
				rowids[N] = fans[goffset+N];
			type = graph[i].type;
			switch (type) {
				case 0: break;
				case INPT:
						pi = piNumber;
						val = *(input+(pi+iwidth*tid));
						if (pass > 1) {
							row[i] = stable[(row[i] > 1)][val];
//							DPRINT("row[%d] = %d \n",i, row[i]);
						} else {
							row[i] = val;
						}
						piNumber++;
						break;
				default: 
						// we're guaranteed at least one fanin per 
						// gate if not on an input.
						if (graph[i].type != NOT) {
							val = row[rowids[0]];
						} else {
							val = notl[row[rowids[0]]];
						}
						j = 1;
						while (j < nfi) {
							r = row[rowids[j]]; 
							switch(type) {
								case XOR:
									val = xor2[val][r];
								case XNOR:
									val = xnor2[val][r];
								case OR:
									val = or2[val][r];
								case NOR:
									val = nor2[val][r];
								case AND:
									val = and2[val][r];
								case NAND:
									val = nand2[val][r];
							}
							j++;
						}
						if (pass > 1 && type != FROM && type != BUFF) {
							row[i] = stable[(row[i] > 1)][val];
						} else {
							row[i] = val;
						}
			}
		}
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
			cpuSimulate(dgraph.data, results.data, inputs.data, fan, inputs.width, results.width,results.height,pass, j);
		}
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
	for (int i = 0; i < height*width; i++) {
		assert(input[i] < 2);
//		DPRINT("%d ",input[i]);
	}
//	DPRINT("\n");
	memcpy(tgt, input,sizeof(int)*(width));
	memcpy(input, input+width,sizeof(int)*(width)*(height-1));
	memcpy(input+(height-1)*(width),tgt, sizeof(int)*(width));
	for (int i = 0; i < height*width; i++) {
		assert(input[i] < 2);
//		DPRINT("%d ",input[i]);
	}
//	DPRINT("\n");
	free(tgt);
}
float cpuMarkPaths(ARRAY2D<int> results, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph,  int* fan) {
	float elapsed = 0.0;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int i = 0; i < results.height; i++) {
//		DPRINT("Running TID: %d ", i);
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
			cpuMerge(i,j,input.data, *mergeresult, input.width);
		}
	memcpy(*mergeresult, input.data, input.bwidth());
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
//	DPRINT("rw %d, rh %d, gw %d, gh %d\n", results.width, results.height, dgraph.width, dgraph.height);
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
void debugCpuSimulationOutput(ARRAY2D<int> results, int pass = 1) {
#ifndef NDEBUG
	int *lvalues, *row;
	DPRINT("Post-simulation device results, pass %d:\n\n", pass);
	DPRINT("Line:   \t");
	for (int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (int r = 0;r < results.height; r++) {
		lvalues = (int*)malloc(results.bwidth());
		row = (int*)((char*)results.data + r*results.bwidth()); // get the current row?
		memcpy(lvalues,row,results.bwidth());
		DPRINT("%s %d:\t", pass > 1 ? "Vector" : "Pattern",r);
		for (int i = 0; i < results.width; i++) {
			switch(lvalues[i]) {
				case S0:
					DPRINT("S0 "); break;
				case S1:
					DPRINT("S1 "); break;
				case T0:
					DPRINT("T0 "); break;
				case T1:
					DPRINT("T1 "); break;
				default:
					DPRINT("%2d ", lvalues[i]); break;
			}
		}
		DPRINT("\n");
		free(lvalues);
	}
#endif
}
