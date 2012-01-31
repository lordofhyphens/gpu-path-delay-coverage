#include "serial.h"

void cpuSimulateP1(const Circuit& ckt, char* pi, char* sim,size_t pi_pitch, size_t pattern);
void cpuSimulateP2(const Circuit& ckt, char* pi, char* sim,size_t pi_pitch, size_t pattern);
void cpuMark(const Circuit& ckt, char* sim, char* mark);
void cpuCover(const Circuit& ckt, char* mark, char* hist, int* cover, int* covered);
inline void cpuMerge(const Circuit& ckt, char* in, char* hist) { for (int i = 0; i < ckt.size(); i++) { hist[i] |= in[i];} }

float serial(Circuit& ckt, CPU_Data& input) {
    float total = 0.0, elapsed;
    timespec start, stop;
    char* simulate;
    char* mark; 
    char* merge = new char[ckt.size()];
    int* cover = new int[ckt.size()];
    int* coverage = new int;
    *coverage = 0;
    for (unsigned int pattern = 0; pattern < input.width(); pattern+=2) {
        simulate = new char[ckt.size()];
        mark = new char[ckt.size()];
        for (int i = 0; i < ckt.size(); i++) {
            simulate[i] = 0;
            mark[i] = 0;
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        // simulate pattern 1
        //std::cerr << "Serial Simulate P1" << std::endl;

        cpuSimulateP1(ckt, input.cpu(), simulate, input.pitch(),pattern);
        // simulate pattern 2
        //std::cerr << "Serial Simulate P2" << std::endl;
        cpuSimulateP2(ckt, input.cpu(), simulate, input.pitch(),pattern+1);
        // mark
        //std::cerr << "Mark" << std::endl;
        cpuMark(ckt, simulate, mark);
        // calculate coverage against all previous runs
        //std::cerr << "Cover" << std::endl;
        cpuCover(ckt, mark, merge, cover, coverage);
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

void cpuSimulateP1(const Circuit& ckt, char* pi, char* sim, size_t pi_pitch, size_t pattern) {
    for (int g = 0; g < ckt.size(); g++) {
        const NODEC gate = ckt.at(g);
        char val=0;
        switch(gate.typ) {
            case INPT:
                val = REF2D(char,pi,pi_pitch,pattern, g);
                break;
            default:
                if (gate.typ != NOT) {
                    val = sim[gate.fin.at(0).second];
                } else {
                    val = !sim[gate.fin.at(0).second];
                }
                int j = 1;
                while (j < gate.nfi) {
                    //std::cerr << __LINE__ << std::endl;
                    switch(gate.typ) {
                        case AND: 
                            val = !(val & sim[gate.fin.at(j).second]);
                            //std::cerr << __LINE__ << std::endl;
                            break;
                        case NAND:
                            val = !(val & sim[gate.fin.at(j).second]);
                            //std::cerr << __LINE__ << std::endl;
                            break;
                        case OR:
                            val = (val | sim[gate.fin.at(j).second]);
                            //std::cerr << __LINE__ << std::endl;
                            break;
                        case NOR: 
                            val = !(val | sim[gate.fin.at(j).second]);
                            //std::cerr << __LINE__ << std::endl;
                            break;
                        case XOR: 
                            val = (val ^ sim[gate.fin.at(j).second]);
                            //std::cerr << __LINE__ << std::endl;
                            break;
                        case XNOR:
                            val = !(val ^ sim[gate.fin.at(j).second]);
                            //std::cerr << __LINE__ << std::endl;
                            break;
                    }
                    j++;
                }
        }
        sim[g] = val;
        DPRINT("Pass 1 %d, %d\n", g, sim[g]);
    }
}
void cpuSimulateP2(const Circuit& ckt, char* pi, char* sim,size_t pi_pitch, size_t pattern) {
	int stable[2][2] = {{S0, T1}, {T0, S1}};
    for (int g = 0; g < ckt.size(); g++) {
        const NODEC gate = ckt.at(g);
        int val;
        switch(gate.typ) {
            case INPT:
                val = REF2D(char,pi,pi_pitch,pattern, g);
                break;
            default:
                if (gate.typ != NOT) {
                    val = sim[(int)gate.fin.at(0).second];
                } else {
                    val = !sim[(int)gate.fin.at(0).second];
                }
                int j = 1;
                while (j < gate.nfi) {
                    switch(gate.typ) {
                        case AND: 
                            val = !(val & sim[gate.fin.at(j).second]);
                            break;
                        case NAND:
                            val = !(val & sim[gate.fin.at(j).second]);
                            break;
                        case OR:
                            val = (val | sim[gate.fin.at(j).second]);
                            break;
                        case NOR: 
                            val = !(val | sim[gate.fin.at(j).second]);
                            break;
                        case XOR: 
                            val = (val ^ sim[gate.fin.at(j).second]);
                            break;
                        case XNOR:
                            val = !(val ^ sim[gate.fin.at(j).second]);
                            break;
                    }
                    j++;
                }
        }
        sim[g] = stable[val][(int)sim[g]];
        DPRINT("Pass 2 %d, %d\n", g, sim[g]);
    }

}
void cpuMark(const Circuit& ckt, char* sim, char* mark) {
    char resultCache, fin, cache, rowCache;
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
				val = (resultCache > 0) && (rowCache > 1);
				f = val || (sim[gate.fin.at(0).second] > 0);
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
						cache = AND_OUT(sim[gate.fin.at(fin1).second], sim[gate.fin.at(fin2).second] );
						pass += (cache > 1);
						tmp = tmp && (cache > 0);
						if (gate.nfi > 2) {
                            cache = AND_IN(sim[gate.fin.at(fin1).second], sim[gate.fin.at(fin2).second] );
							fin = cache && fin && prev;
						}
					}
                    mark[gate.fin.at(fin1).second] = fin;
				}
				resultCache= val && tmp && (pass < gate.nfi) && prev;
				break;
			case OR:
			case NOR:
				for (fin1 = 0; fin1 < gate.nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < gate.nfi; fin2++) {
						if (fin1 == fin2) continue;
						cache = OR_OUT(sim[gate.fin.at(fin1).second], sim[gate.fin.at(fin2).second] );
						pass += (cache > 1);
						tmp = tmp && (cache > 0);

						if (gate.nfi > 2) {
                            cache = OR_IN(sim[gate.fin.at(fin1).second], sim[gate.fin.at(fin2).second] );
							fin = cache && fin && prev;
						}

					}
                    mark[gate.fin.at(fin1).second] = fin;
				}
				resultCache= val && tmp && (pass <= gate.nfi) && prev;
				break;
			case XOR:
			case XNOR:
				for (fin1 = 0; fin1 < gate.nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < gate.nfi; fin2++) {
						if (fin1 == fin2) continue;
                        cache = XOR_OUT(sim[gate.fin.at(fin1).second], sim[gate.fin.at(fin2).second] );
						pass += (cache > 1);
						tmp = tmp && (cache > 0);
						if (gate.nfi > 2) {
                            cache = XOR_IN(sim[gate.fin.at(fin1).second], sim[gate.fin.at(fin2).second] );
							fin = cache && fin && prev;
						}
					}
                    mark[gate.fin.at(fin1).second] = fin;
				}
				resultCache = val && tmp && (pass <= gate.nfi) && prev;
				break;
			default:
				// if there is a transition that will propagate, set = to some positive #?
				break;
		}
		// stick the contents of resultCache into the results array
		mark[g] = resultCache;
        DPRINT("%d = %d\n",g,mark[g]);
	}
}
#define FREF(AR, GATE, FIN, REF) ((AR[GATE.FIN.at(REF).second]))
#define FADDR(GATE,FIN,REF) (GATE.FIN.at(REF).second)
#define NOTMARKED(MARK, HIST, GATE) ((MARK[GATE] > HIST[GATE]))
void cpuCover(const Circuit& ckt, char* mark, char* hist, int* cover, int* covered) {
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
