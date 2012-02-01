#include "serial.h"

void cpuSimulateP1(const Circuit& ckt, char* pi, int* sim,size_t pi_pitch, size_t pattern);
void cpuSimulateP2(const Circuit& ckt, char* pi, int* sim,size_t pi_pitch, size_t pattern);
void cpuMark(const Circuit& ckt, int* sim, int* mark);
void cpuCover(const Circuit& ckt, int* mark, int* hist, int* cover, int* covered);
inline void cpuMerge(const Circuit& ckt, int* in, int* hist) { for (int i = 0; i < ckt.size(); i++) { hist[i] |= in[i];} }

void debugPrintSim(const Circuit& ckt, int* in, int pattern, int type) {
	for (int i = 0; i < ckt.size(); i++) {
		switch (type) {
			case 2:
				switch(in[i]) {
					case S0:
						DPRINT("S0 "); break;
					case S1:
						DPRINT("S1 "); break;
					case T0:
						DPRINT("T0 "); break;
					case T1:
						DPRINT("T1 "); break;
					default:
						DPRINT("%2d ", in[i]);
				} break;
			case 3: 
				DPRINT("%2c ", (in[i] > 0 ? 'Y' : 'N')); break;
			default:
				DPRINT("%2d ", in[i]);
		}
	}
	DPRINT("\n");
}

float serial(Circuit& ckt, CPU_Data& input) {
    float total = 0.0, elapsed;
    timespec start, stop;
    int* simulate;
    int* mark; 
    int* merge = new int[ckt.size()];
    int* cover = new int[ckt.size()];
    int* coverage = new int;
    *coverage = 0;
	std::cerr << "CPU results:" << std::endl;
	std::cerr << "    Line: ";
	for (int i = 0; i < ckt.size(); i++) { 
		DPRINT("%2d ", i);
	}
	std::cerr << std::endl;

    for (unsigned int pattern = 0; pattern < input.width(); pattern++) {
        simulate = new int[ckt.size()];
        mark = new int[ckt.size()];
        for (int i = 0; i < ckt.size(); i++) {
            simulate[i] = 0;
            mark[i] = 0;
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        // simulate pattern 1
        //std::cerr << "Serial Simulate P1" << std::endl;
        cpuSimulateP1(ckt, input.cpu(), simulate, input.pitch(),pattern);
		std::cerr << "Simulate: ";
		debugPrintSim(ckt, simulate,pattern, 2);
        // simulate pattern 2
        //std::cerr << "Serial Simulate P2" << std::endl;
		if (pattern == (input.width()-1))  {
			cpuSimulateP2(ckt, input.cpu(), simulate, input.pitch(),0);
		}
		else {
			cpuSimulateP2(ckt, input.cpu(), simulate, input.pitch(),pattern+1);
		}
		std::cerr << "Simulate: ";
		debugPrintSim(ckt, simulate,pattern, 2);
        // mark
        //std::cerr << "Mark" << std::endl;
        cpuMark(ckt, simulate, mark);
		std::cerr << "    Mark: ";
		debugPrintSim(ckt, mark,pattern, 3);
        // calculate coverage against all previous runs
        //std::cerr << "Cover" << std::endl;
        cpuCover(ckt, mark, merge, cover, coverage);
//		std::cerr << "   Cover: ";
//		debugPrintSim(ckt, cover,pattern, 4);
        // merge mark to history
        //std::cerr << "Merge" << std::endl;
        cpuMerge(ckt, mark, merge);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        elapsed = floattime(diff(start, stop));
        total += elapsed;
        delete mark;
        delete simulate;
    }
    DPRINT("Serial Coverage: %d\n", *coverage);
    delete coverage;
    return total;
}

void cpuSimulateP1(const Circuit& ckt, char* pi, int* sim, size_t pi_pitch, size_t pattern) {
    for (int g = 0; g < ckt.size(); g++) {
        const NODEC gate = ckt.at(g);
        int val=0;
        switch(gate.typ) {
            case INPT:
                val = REF2D(char,pi,pi_pitch,pattern,g); break;
			case FROM:
				val = FREF(sim,gate,fin,0); break;
            default:
                if (gate.typ != NOT) {
                    val = FREF(sim,gate,fin,0);
                } else {
                    val = (FREF(sim,gate,fin,0) != 1);
                }
                int j = 1;
                while (j < gate.nfi) {
                    switch(gate.typ) {
                        case AND: 
                            val = (val & FREF(sim,gate,fin,j));
                            break;
                        case NAND:
                            val = !(val & FREF(sim,gate,fin,j));
                            break;
                        case OR:
                            val = (val | FREF(sim,gate,fin,j));
                            break;
                        case NOR: 
                            val = !(val | FREF(sim,gate,fin,j));
                            break;
                        case XOR: 
                            val = (val ^ FREF(sim,gate,fin,j));
                            break;
                        case XNOR:
                            val = !(val ^ FREF(sim,gate,fin,j));
                            break;
                    }
                    j++;
                }
        }
        sim[g] = val;
    }
}
void cpuSimulateP2(const Circuit& ckt, char* pi, int* sim,size_t pi_pitch, size_t pattern) {
	int stable[2][2] = {{S0, T1}, {T0, S1}};
    for (int g = 0; g < ckt.size(); g++) {
        const NODEC gate = ckt.at(g);
        int val;
        switch(gate.typ) {
            case INPT: val = REF2D(char,pi,pi_pitch,pattern, g); break;
			case FROM: val = BIN(FREF(sim,gate,fin,0)); break;
            default:
                if (gate.typ != NOT) {
                    val = BIN(FREF(sim,gate,fin,0));
                } else {
                    val = (BIN(FREF(sim,gate,fin,0)) != 1);
                }
                int j = 1;
                while (j < gate.nfi) {
                    switch(gate.typ) {
                        case AND: val = (BIN(val) & BIN(FREF(sim,gate,fin,j))); break;
                        case NAND: val = !(BIN(val) & BIN(FREF(sim,gate,fin,j))); break;
                        case OR: val = (BIN(val) | BIN(FREF(sim,gate,fin,j))); break;
                        case NOR: val = !(BIN(val) | BIN(FREF(sim,gate,fin,j))); break;
                        case XOR: val = (BIN(val) ^ BIN(FREF(sim,gate,fin,j))); break;
                        case XNOR: val = !(BIN(val) ^ BIN(FREF(sim,gate,fin,j))); break;
                    }
                    j++;
                }
        }
//		std::cerr << gate.name << " sim[" << g<< "] " << sim[g] << " val " << val << " stable: " << stable[sim[g]][val];
		sim[g] = stable[sim[g]][val];
//		std::cerr << " sim["<<g<<"]: " << sim[g] << std::endl;
    }

}

void cpuMark(const Circuit& ckt, int* sim, int* mark) {
    int resultCache, fin, cache, rowCache;
    int tmp = 1, pass = 0, fin1 = 0, fin2 = 0, val, f,prev;
    for (int g = ckt.size()-1; g >= 0;g--) {
        const NODEC gate = ckt.at(g);
        resultCache = mark[g];
        rowCache = sim[g];
		val = (rowCache > 1);
        if (gate.po > 0) {
           resultCache = val; 
           prev = val;
        } else { prev = resultCache; }
        switch(gate.typ) {
            case FROM:
				val = ((resultCache > 0) && (rowCache > 1));
				std::cerr << "FROM: " << g << " " << val << std::endl;
				f = val || (FREF(mark,gate,fin,0) > 0);
				FREF(mark,gate,fin,0) |= f;
                resultCache = val;
                break;
            case BUFF:
            case NOT:
				val = NOT_IN(rowCache) && prev;
				mark[g] = val;
				resultCache = val;
                break;
				// For the standard gates, setting three values -- both the
				// input lines and the output line.  rowCache[threadIdx.x][i]-1 is the
				// transition on the output, offset to make the texture
				// calculations correct because there are 4 possible values
				// rowCache[threadIdx.x][i] can take: 0, 1, 2, 3.  0, 1 are the same, as are
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
				for (fin1 = 1; fin1 < gate.nfi; fin1++) {
					fin = 1;
					for (fin2 = 1; fin2 < gate.nfi; fin2++) {
						if (fin1 == fin2) continue;
						cache = AND_OUT(FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2) );
						pass += (cache > 1);
						tmp = tmp && (cache > 0);
						if (gate.nfi > 2) {
                            cache = AND_IN(FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2) );
							fin = cache && fin && prev;
						}
					}
					std::cerr << "Gate " << std::endl;
                    FREF(mark,gate,fin,fin1) = fin;
				}
				resultCache= val && tmp && (pass < gate.nfi) && prev;
				break;
			case OR:
			case NOR:
				for (fin1 = 0; fin1 < gate.nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < gate.nfi; fin2++) {
						if (fin1 == fin2) continue;
						cache = OR_OUT(FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2) );
						pass += (cache > 1);
						tmp = tmp && (cache > 0);

						if (gate.nfi > 2) {
                            cache = OR_IN(FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2)  );
							fin = cache && fin && prev;
						}

					}
                    FREF(mark,gate,fin,fin1) = fin;
				}
				resultCache = val && tmp && (pass <= gate.nfi) && prev;
				break;
			case XOR:
			case XNOR:
				for (fin1 = 0; fin1 < gate.nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < gate.nfi; fin2++) {
						if (fin1 == fin2) continue;
                        cache = XOR_OUT(FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2));
						pass += (cache > 1);
						tmp = tmp && (cache > 0);
						if (gate.nfi > 2) {
                            cache = XOR_IN(FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2) );
							fin = cache && fin && prev;
						}
					}
                    FREF(mark,gate,fin,fin1) = fin;
				}
				resultCache = val && tmp && (pass <= gate.nfi) && prev;
				break;
			default:
				// if there is a transition that will propagate, set = to some positive #?
				break;
		}
		// stick the contents of resultCache into the results array
		mark[g] = resultCache;
	}
}
void cpuCover(const Circuit& ckt, int* mark, int* hist, int* cover, int* covered) {
    // cover is the coverage ints we're working with for this pass.
    // mark is the fresh marks
    // hist is the history of the mark status of all lines.
    int* hist_cover = new int[ckt.size()];
    int val = 0;
    for (int g = ckt.size()-1; g >= 0; g--) {
        const NODEC& gate = ckt.at(g);
        if (gate.po) {
            cover[g] = (mark[g] > 0) && (hist[g] <= 0);
            hist_cover[g] = (hist[g] > 0);
        }
        if (gate.typ == FROM) {
            val = cover[g]*(NOTMARKED(mark,hist,g));

            FREF(cover,gate,fin,0) += val;
            FREF(hist_cover,gate,fin,0) += hist_cover[g];
        } else {
            for (int i = 0; i < gate.nfi; i++) {
                FREF(cover,gate,fin,i) = (NOTMARKED(mark,hist,g))*cover[g];
                FREF(hist_cover,gate,fin,i) = (!NOTMARKED(mark,hist,g))*hist_cover[g];
            }
        }
        if (gate.typ == INPT) {
            *covered = *covered + cover[g];
        }
    }
    delete hist_cover;
}
