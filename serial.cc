#include "serial.h"

void cpuSimulateP1(const Circuit& ckt, char* pi, int* sim,size_t pi_pitch, size_t pattern);
void cpuSimulateP2(const Circuit& ckt, char* pi, int* sim,size_t pi_pitch, size_t pattern);
void cpuMark(const Circuit& ckt, int* sim, int* mark);
void cpuCover(const Circuit& ckt, int* mark, int* hist, int* cover, int* hist_cover, int* covered);
inline void cpuMerge(const Circuit& ckt, int* in, int* hist) { for (int i = 0; i < ckt.size(); i++) { hist[i] = hist[i] | in[i];} }

void debugPrintSim(const Circuit& ckt, int* in, int pattern, int type, std::ostream& ofile) {
	ofile << "Vector " << pattern << ":\t";
	for (int i = 0; i < ckt.size(); i++) {
		switch (type) {
			case 2:
				switch(in[i]) {
					case S0:
						ofile  << std::setw(OUTJUST+1) << "S0 "; break;
					case S1:
						ofile  << std::setw(OUTJUST+1) << "S1 "; break;
					case T0:
						ofile  << std::setw(OUTJUST+1) << "T0 "; break;
					case T1:
						ofile  << std::setw(OUTJUST+1) << "T1 "; break;
					default:
						ofile << std::setw(OUTJUST) << (int)in[i] << " "; break;
				} break;
			case 3: 
				ofile << std::setw(OUTJUST) << (in[i] > 0 ? 'Y' : 'N') << " "; break;
			default:
				ofile << std::setw(OUTJUST) << (int)in[i] << " "; break;
		}
	}
	ofile << std::endl;
}

float serial(Circuit& ckt, CPU_Data& input) {
	std::ofstream s1file("serialsim-p1.log", std::ios::out);
	std::ofstream s2file("serialsim-p2.log", std::ios::out);
	std::ofstream mfile("serialmark.log", std::ios::out);
	std::ofstream cfile("serialcover.log", std::ios::out);
    float total = 0.0, elapsed;
    timespec start, stop;
    int* simulate;
    int* mark; 
    int* merge = new int[ckt.size()];
    int* cover = new int[ckt.size()];
    int* hist_cover;
	int* coverage = new int;
    *coverage = 0;
	
/*	std::cerr << "CPU results:" << std::endl;
	std::clog << "Line:   \t";
	for (int i = 0; i < ckt.size(); i++) { 
		DPRINT("%3d ", i);
	}
	std::clog << std::endl;
*/
	for (int i = 0; i < ckt.size(); i++) {
		merge[i] = 0;
	}
    for (unsigned int pattern = 0; pattern < input.width(); pattern++) {
        simulate = new int[ckt.size()];
        mark = new int[ckt.size()];
		cover = new int[ckt.size()];
		hist_cover = new int[ckt.size()];
        for (int i = 0; i < ckt.size(); i++) {
            simulate[i] = 0;
            mark[i] = 0;
			cover[i] = 0;
			hist_cover[i] = 0;
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        // simulate pattern 1
        //std::cerr << "Serial Simulate P1" << std::endl;
        cpuSimulateP1(ckt, input.cpu().data, simulate, input.cpu().pitch,pattern);
		//std::cerr << "Simulate: ";
		debugPrintSim(ckt, simulate,pattern, 2, s1file);
        // simulate pattern 2
        //std::cerr << "Serial Simulate P2" << std::endl;
		if (pattern == (input.width()-1))  {
			cpuSimulateP2(ckt, input.cpu().data, simulate, input.cpu().pitch,0);
		}
		else {
			cpuSimulateP2(ckt, input.cpu().data, simulate, input.cpu().pitch,pattern+1);
		}
		//std::cerr << "Simulate: ";
		debugPrintSim(ckt, simulate,pattern, 2, s2file);
        // mark
        //std::cerr << "Mark" << std::endl;
        cpuMark(ckt, simulate, mark);
		//std::cerr << "    Mark: ";
		debugPrintSim(ckt, mark,pattern, 3, mfile);
        // calculate coverage against all previous runs
        cpuCover(ckt, mark, merge, hist_cover, cover,coverage);
		//std::cerr << "   Cover: ";
		//debugPrintSim(ckt, cover,pattern, 4, cfile);
        // merge mark to history
        //std::cerr << "Merge" << std::endl;
        cpuMerge(ckt, mark, merge);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        elapsed = floattime(diff(start, stop));
//		std::cerr << "Pass " << pattern << " time: " << elapsed<<std::endl;
        total += elapsed;
        delete mark;
        delete simulate;
		delete cover;
		delete hist_cover;
    }
    DPRINT("Serial Coverage: %d\n", *coverage);
    delete coverage;
	s1file.close();
	s2file.close();
	mfile.close();
	cfile.close();
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
		sim[g] = stable[sim[g]][val];
    }

}

void cpuMark(const Circuit& ckt, int* sim, int* mark) {
	int and2_output_prop[16]= {0,0,0,0,0,2,1,1,0,1,1,0,0,1,0,1};
	int and2_input_prop[16] = {0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1};
	int or2_output_prop[16] = {2,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1};
	int or2_input_prop[16]  = {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	int xor2_input_prop[16] = {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	int xor2_output_prop[16]= {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};

    int resultCache, fin, cache, rowCache;
    int tmp = 1, pass = 0, fin1 = 0, fin2 = 0, val, prev;
    for (int g = ckt.size()-1; g >= 0;g--) {
        const NODEC gate = ckt.at(g);
        resultCache = mark[g];
        rowCache = sim[g];
		val = (rowCache > 1);
        if (gate.po > 0) {
           resultCache = val; 
           prev = val;
        } else { prev = resultCache; }
		if (gate.nfo > 1) {
			prev = 0;
			resultCache = 0;
			for (int i = 0; i < gate.nfo; i++) {
				resultCache |= FREF(mark,gate,fot,i);
			}
			prev = resultCache;
		}
        switch(gate.typ) {
            case FROM: break;
            case BUFF:
            case NOT:
				val = NOT_IN(rowCache) && prev;
				FREF(mark,gate,fin,0) = val;
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
				for (fin1 = 0; fin1 < gate.nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < gate.nfi; fin2++) {
						if (fin1 == fin2) continue;
						cache = REF2D(int,and2_output_prop,sizeof(int)*4,FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2));
						pass += (cache > 1);
						tmp = tmp && (cache > 0);
						if (gate.nfi > 1) {
							cache = REF2D(int,and2_input_prop,sizeof(int)*4,FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2));
							fin = cache && fin && prev;
						}
					}
//					std::cerr << "mark[" << FADDR(gate,fin,fin1) <<  "] = "<< fin<< std::endl;
                    FREF(mark,gate,fin,fin1) = fin;
				}
//				resultCache= val && tmp && (pass <= gate.nfi) && prev;
//				std::clog << "mark[" << g << "]: " << resultCache << std::endl;
				break;
			case OR:
			case NOR:
				for (fin1 = 0; fin1 < gate.nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < gate.nfi; fin2++) {
						if (fin1 == fin2) continue;
						cache = REF2D(int,or2_output_prop,sizeof(int)*4,FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2));
						pass += (cache > 1);
						tmp = tmp && (cache > 0);

						if (gate.nfi >= 2) {
							cache = REF2D(int,or2_input_prop,sizeof(int)*4,FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2));
							fin = cache && fin && prev;
						}

					}
                    FREF(mark,gate,fin,fin1) = fin;
				}
//				resultCache = val && tmp && (pass <= gate.nfi) && prev;
				break;
			case XOR:
			case XNOR:
				for (fin1 = 0; fin1 < gate.nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < gate.nfi; fin2++) {
						if (fin1 == fin2) continue;
						cache = REF2D(int,xor2_output_prop,sizeof(int)*4,FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2));
						pass += (cache > 1);
						tmp = tmp && (cache > 0);
						if (gate.nfi >= 2) {
							cache = REF2D(int,xor2_input_prop,sizeof(int)*4,FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2));
							fin = cache && fin && prev;
						}
					}
                    FREF(mark,gate,fin,fin1) = fin;
				}
//				resultCache = val && tmp && (pass <= gate.nfi) && prev;
				break;
			default:
				// if there is a transition that will propagate, set = to some positive #?
				break;
		}
		// stick the contents of resultCache into the results array
		mark[g] = resultCache;
	}
}
void cpuCover(const Circuit& ckt, int* mark, int* hist, int* hist_cover, int* cover, int* covered) {
    // cover is the coverage ints we're working with for this pass.
    // mark is the fresh marks
    // hist is the history of the mark status of all lines.
    for (int g = ckt.size()-1; g >= 0; g--) {
        const NODEC& gate = ckt.at(g);
        if (gate.po == true) {
            cover[g] = 0;
            hist_cover[g] = (mark[g] > 0);
        }
		if (gate.nfo > 1) {
			for (int i = 0; i < gate.nfo; i++) {
				cover[g] += FREF(cover,gate,fot,i);
				hist_cover[g] += FREF(hist_cover,gate,fot,i);
			}
		}
		if (gate.typ != FROM) {
			cover[g] = (NOTMARKED(mark,hist,g))*(cover[g]+hist_cover[g]);
			hist_cover[g] = (NOTMARKED(mark,hist,g) ? 0 : hist_cover[g]);
            for (int i = 0; i < gate.nfi; i++) {
				FREF(cover,gate,fin,i) = cover[g];
				FREF(hist_cover,gate,fin,i) = hist_cover[g];
            }
        } 
		if (gate.typ == INPT)
            *covered = *covered + cover[g];
		//hist[g] |= mark[g];
	}
}
