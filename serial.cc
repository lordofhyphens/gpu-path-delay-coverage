#include "serial.h"
#include "iscas.h"
#include <ctime>
#include <time.h>
#include <unistd.h>

void cpuMerge(int h, int w, int* input, int* results, int width) {
//	int merge[2][2] = {{0,1},{1,1}};
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
	int and2_output_prop[4][4]= {{0,0,0,0},{0,0,1,1},{1,1,1,1},{0,1,1,1}};
	int and2_input_prop[4][4] = {{0,0,0,0},{0,0,1,1},{0,0,1,0},{0,0,1,1}};
	int or2_output_prop[4][4] = {{2,0,1,1},{0,0,0,0},{1,0,1,1},{1,0,1,1}};
	int or2_input_prop[4][4]  = {{0,0,1,1},{0,0,0,0},{0,0,1,1},{0,0,0,1}};
	int xor2_output_prop[4][4] = {{2,0,1,1},{0,0,0,0},{1,0,1,1},{1,0,1,1}};
	int xor2_input_prop[4][4]  = {{0,0,1,1},{0,0,0,0},{0,0,1,1},{0,0,0,1}};
//	int from_prop[16]      =  {0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1};
	int inpt_prop[2][4] = {{0,0,0,0},{0,0,1,1}};
	int nfi, goffset,val;
	int rowids[50];
	char cache[1]; // needs to be 2x # of threads being run
	int tmp = 1, pass = 0, fin1 = 0, fin2 = 0,fin = 1, type;
	int *rowResults, *row;
	if ((unsigned int)tid < height) {
		cache[0] = 0;
		row = (int*)((char*)results + tid*(width)*sizeof(int));
		rowResults = (int*)malloc(sizeof(int)*width);
		for (int i = 0; i < ncount; i++) {
			rowResults[i] = 0;
		}
		for (int i = ncount-1; i >= 0; i--) {
			nfi = node[i].nfi;
			type = node[i].type;
			if (0 == 0) {
				goffset = node[i].offset;
				// preload all of the fanin line #s for this gate to shared memory.
				for (int j = 0; j < nfi;j++) {
					rowids[j] = fans[goffset+j];
				}
			}
			// switching based on value causes divergence, switch based on node type.
			val = (row[i] > 1);
			if (node[i].po) {
				rowResults[i] = val;
			}
			switch(type) {
				case FROM:
					// For FROM, only set the "input" line if it hasn't already
					// been set (otherwise it'll overwrite the decision of
					// another system somewhere else.

					val = (rowResults[i] > 0 && row[rowids[0]] > 1);
					rowResults[rowids[0]] = val || rowResults[i];
					rowResults[i] =  val;
					break;
				case BUFF:
				case NOT:
					val = inpt_prop[row[rowids[0]]][rowResults[i]] && rowResults[i];
					rowResults[rowids[0]] = val;
					rowResults[i] = val;
					break;
					// For the standard gates, setting three values -- both the
					// input lines and the output line.  row[i]-1 is the
					// transition on the output, offset to make the texture
					// calculations correct because there are 4 possible values
					// row[i] can take: 0, 1, 2, 3.  0, 1 are the same, as are
					// 2,3, so we subtract 1 and clamp to an edge if we
					// overflow.
					// 0 becomes -1 (which clamps to 0)
					// 1 becomes 0
					// 2 becomes 1
					// 3 becomes 2 (which clamps to 1)
					// There's only 2 LUTs for each gate type. The input LUT
					// checks for path existance through the first input, so we
					// call it twice with the inputs reversed to check both
					// paths.

				case NAND:
				case AND:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache[0] = and2_output_prop[row[rowids[fin1]]][row[rowids[fin2]]];
							pass += (cache[0] > 1);
							tmp = tmp && ((int)cache[0] > 0);
						}
					}
					rowResults[i] = val && tmp && (pass <= nfi);
					break;
				case OR:
				case NOR:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						fin = 1;
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache[0] = or2_output_prop[row[rowids[fin1]]][row[rowids[fin2]]];
							pass += (cache[0] > 1);
							tmp = tmp && ((int)cache[0] > 0);
						}
					}
					rowResults[i] = val && tmp && (pass <= nfi);
					break;
				case XOR:
				case XNOR:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache[0] = xor2_output_prop[row[rowids[fin1]]][row[rowids[fin2]]];
							pass += (cache[0] > 1);
							tmp = tmp && ((int)cache[0] > 0);
						}
					}
					rowResults[i] = val && tmp && (pass <= nfi);
					break;
				default:
					// if there is a transition that will propagate, set = to some positive #?
					break;
			}
			switch(type) {
				case AND:
				case NAND:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						fin = 1;
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache[0] = and2_input_prop[row[rowids[fin1]]][row[rowids[fin2]]];
							fin = cache[0] && fin;
						}
						rowResults[rowids[fin1]] = fin && rowResults[i];
					}
					break;
				case OR:
				case NOR:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						fin = 1;
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache[0] = or2_input_prop[row[rowids[fin1]]][row[rowids[fin2]]];
							fin = cache[0] && fin;
						}
						rowResults[rowids[fin1]] = fin && rowResults[i];
					}
				case XOR:
				case XNOR:
					for (fin1 = 0; fin1 < nfi; fin1++) {
						fin = 1;
						for (fin2 = 0; fin2 < nfi; fin2++) {
							if (fin1 == fin2) continue;
							cache[0] = xor2_input_prop[row[rowids[fin1]]][row[rowids[fin2]]];
							fin = cache[0] && fin;
						}
						rowResults[rowids[fin1]] = fin && rowResults[i];
					}
					break;
				default:
					;;

			}
		}
		// replace our working set to save memory.
		for (unsigned int i = 0; i < width; i++) {
			row[i] = rowResults[i];
		}
		free(rowResults);
	}
}

int cpuCountCoverage(const int toffset, const unsigned int tid, int *results, int *history, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
	int nfi, goffset, count = 0;
	int *row, *historyRow;
	int *current, *historyCount;
	int rowids[50]; // handle up to fanins of 50 /
	if (tid < height) {
		row = (int*)((char*)results + tid*(width)*sizeof(int));
		if (tid == 0) {
			historyRow = (int*)malloc(sizeof(int)*width);
			memset(historyRow, 0, sizeof(int)*width);
		} else {
			historyRow = history;
		}
		current = (int*)malloc(sizeof(int)*width);
		historyCount = (int*)malloc(sizeof(int)*width);
		for (int i = 0; i < ncount; i++) {
			current[i] = 0;
			historyCount[i] = 0;
		}
		for (int i = ncount-1; i >= 0; i--) {
			nfi = node[i].nfi;
			if (tid == 0) {
				goffset = node[i].offset;
				// preload all of the fanin line #s for this gate for less faffing about.
				for (int j = 0; j < nfi;j++) {
					rowids[j] = fans[goffset+j];
				}
			}
			if (node[i].po) {
				current[i] = (row[i] > historyRow[i]); // only set = 1 if there's a new line here
				historyCount[i] = historyRow[i];
			}
			switch(node[i].type) {
				case 0: continue;
				case FROM:
						// Add the current fanout count to the fanin if this line is marked (and the history isn't).
						current[rowids[0]] += current[i]*(row[rowids[0]] > historyRow[rowids[0]]);
						historyCount[rowids[0]] += historyCount[i]*(historyRow[rowids[0]]);
						break;
				case INPT:
						continue;
				default: 
						for (int fin = 0; fin < node[i].nfi; fin++) {
							// if the fanout total is 0 but this line is marked (and the history isn't), add a path to the count.
							// If the fanout total is > 1 and this line is marked (and the history isn't), assign the fanout total to the fanins.
							historyCount[rowids[fin]] += (historyRow[rowids[fin]] || historyCount[i] > 1) * historyCount[i];
							current[rowids[fin]] += ((row[rowids[fin]] > historyRow[rowids[fin]]) || current[i] > 1) * current[i] + historyCount[i]*(current[i] == 0 && row[rowids[fin]] > historyRow[rowids[fin]]);
						}

			}
		}
		for (int i = 0; i < ncount; i++) {
			row[i] = current[i];
		}
		for (unsigned int i = 0; i < width; i++)
			if (node[i].type == INPT)
				count += row[i];
		
		free(current);
		free(historyCount);
		if (tid == 0) {
			free(historyRow);
		}
		return count;
	}
	return 0;
}
void cpuSimulate(GPUNODE* graph, int* res, int* input, int* fans, size_t iwidth, size_t width, size_t height, int pass, unsigned int tid) {
	int nand2[4][4] = {{1, 1, 1, 1}, {1, 0, 1, 0}, {1, 1, 1, 1}, {1, 0, 1, 0}};
	int and2[4][4]  = {{0, 0, 0, 0}, {0, 1, 0, 1}, {0, 0, 0, 0}, {0, 1, 0, 1}};
	int nor2[4][4]  = {{1, 0, 1, 0}, {0, 0, 0, 0}, {1, 0, 1, 0}, {0, 0, 0, 0}};
	int or2[4][4]   = {{0, 1, 0, 1}, {1, 1, 1, 1}, {0, 1, 0, 1}, {1, 1, 1, 1}};
	int xnor2[4][4] = {{1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}};
	int xor2[4][4]  = {{0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}};
	int stable[2][2] = {{S0, T1}, {T0, S1}};
	//int from[4] = {0, 0, 1, 1};
	int notl[4] = {1, 0, 1, 0};
	int rowids[1000]; // handle up to fanins of 1000 / 
	int piNumber = 0, pi = 0;
	int *row;
	int goffset, nfi, val = 0, j,type, r;
	if (tid < height) {
		row = res + tid*width; // get the current row?
		for (unsigned int i = 0; i < width; i++) {
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
							row[i] = stable[row[i]][val];  
						} else {
							row[i] = val;
						}
						piNumber++;
						continue;
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
			}
			switch (pass) {
				case 1:
					assert(val < 2);
					row[i] = val;
					assert(row[i] < 2);
					break;
				default:
					if (type != FROM && type != BUFF) {
						assert(row[i] < 2);
						assert(val < 2);
						row[i] = stable[row[i]][val];
					}  else {
						row[i] = val;
					}
			}
		}
	}
}


float cpuRunSimulation(ARRAY2D<int> results, ARRAY2D<int> inputs, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan, int pass) {
	float elapsed = 0.0;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (unsigned int j = 0; j < results.height; j++) {
		cpuSimulate(dgraph.data, results.data, inputs.data, fan, inputs.width, results.width,results.height,pass,j);
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	return elapsed;
}

int* cpuLoad1DVector(int* input, size_t width, size_t height) {
	int *tgt;
	tgt = (int*)malloc(sizeof(int)*(width)*(height));
	memcpy(tgt, input,sizeof(int)*(width)*(height));
	return tgt;
}

int* cpuAllocateResults(size_t width, size_t height) {
	int *tgt;
	tgt = (int*)malloc(sizeof(int)*(width)*(height));
	memset(tgt, 0, sizeof(int)*width*height);
	return tgt;
}

GPUNODE* cpuLoadCircuit(const GPUNODE* graph, int maxid) {
	GPUNODE *devAr;
	devAr = (GPUNODE*)malloc(sizeof(GPUNODE)*(maxid));
	memcpy(devAr, graph, (maxid) * sizeof(GPUNODE));
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
	int* tgt;
	// create a temporary buffer area on the device
	tgt = (int*)malloc(sizeof(int)*(width));
	for (unsigned int i = 0; i < height*width; i++) {
		assert(input[i] < 2);
//		DPRINT("%d ",input[i]);
	}
//	DPRINT("\n");
	memcpy(tgt, input,sizeof(int)*(width));
	memcpy(input, input+width,sizeof(int)*(width)*(height-1));
	memcpy(input+(height-1)*(width),tgt, sizeof(int)*(width));
	for (unsigned int i = 0; i < height*width; i++) {
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
	for (unsigned int i = 0; i < results.height; i++) {
//		DPRINT("Running TID: %d ", i);
		cpuMarkPathSegments(results.data, i, dgraph.data, fan, results.width, results.height, dgraph.width);
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	return elapsed;
}
float cpuMergeHistory(ARRAY2D<int> input, int** mergeresult, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan) {
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	*(mergeresult) = (int*)malloc(sizeof(int)*input.height*input.width);
	for (unsigned int i = 0; i < input.height; i++)
		for (unsigned int j = 0; j< input.width; j++) {
			cpuMerge(i,j,input.data, *mergeresult, input.width);
		}
	memcpy(*mergeresult, input.data, input.bwidth());
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	return elapsed;
}


void cpuSumAll(int toffset, int tid, int *results, int *history, GPUNODE* node, int* fans, size_t width, size_t height, int ncount) {
	int nfi, goffset;
	int *row;
	int sum;
	if (tid < 1) {
		sum = 0;
		for (unsigned int j = 0; j < height; j++) {
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

float cpuCountPaths(ARRAY2D<int> results, GPUNODE* graph, ARRAY2D<GPUNODE> dgraph, int* fan, int* coverage) {
	float elapsed = 0.0;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	int *history = (int*)malloc(sizeof(int)*results.width);
	int *current;
	int cover = 0;
	for (unsigned int i = 0; i < results.width; i++) 
		history[i] = 0;
	for (unsigned int j = 0; j < results.height; j++) {
		current = (int*)((char*)results.data + j*(results.width)*sizeof(int));
		cover += cpuCountCoverage(0, j,results.data, history,dgraph.data, fan, results.width, results.height, dgraph.width);
		for (unsigned int i = 0; i < results.width; i++) 
			history[i] = history[i] | current[i];
	}
	*coverage = cover;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
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
	for (unsigned int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (unsigned int r = 0;r < results.height; r++) {
		row = (int*)((char*)results.data + r*results.bwidth()); // get the current row?
		DPRINT("%s %d:\t","Vector",r);
		for (unsigned int i = 0; i < results.width; i++) {
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
	for (unsigned int i = 0; i < results.width; i++) {
		DPRINT("%2d ", i);
	}
	DPRINT("\n");
	for (unsigned int r = 0;r < results.height; r++) {
		lvalues = (int*)malloc(results.bwidth());
		row = (int*)((char*)results.data + r*results.bwidth()); // get the current row?
		memcpy(lvalues,row,results.bwidth());
		DPRINT("%s %d:\t", pass > 1 ? "Vector" : "Pattern",r);
		for (unsigned int i = 0; i < results.width; i++) {
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
