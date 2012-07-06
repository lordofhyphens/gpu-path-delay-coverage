#include "serial.h"
#include <fstream>
#include <cstdlib>
#include <string>
#include <cstring>
#include <stdint.h>

/*typedef int32_t signed int;
typedef uint32_t unsigned int;
typedef uint8_t unsigned uint8_t;
typedef uint64_t long unsigned int;
*/
void cpuSimulateP1(const Circuit& ckt, uint8_t* pi, uint32_t* sim,size_t pi_pitch, size_t pattern);
void cpuSimulateP2(const Circuit& ckt, uint8_t* pi, uint32_t* sim,size_t pi_pitch, size_t pattern);
void cpuMark(const Circuit& ckt, uint32_t* sim, uint32_t* mark);
void cpuCover(const Circuit& ckt, uint32_t* mark, uint32_t* hist, uint32_t* cover, uint32_t* hist_cover, uint64_t* covered);
inline void cpuMerge(const Circuit& ckt, uint32_t* in, uint32_t* hist) { for (uint32_t i = 0; i < ckt.size(); i++) { hist[i] = hist[i] | in[i];} }

void cpuMergeLog(const Circuit& ckt, uint32_t * in, int32_t* hist, uint32_t p) { 
	for (uint32_t i = 0; i < ckt.size(); i++) { 
		if (hist[i] < 0) { 
			if (in[i] == 1) { 
				hist[i] = p; 
			} 
		} 
	}
}

void debugPrintSim(const Circuit& ckt, uint32_t* in, uint32_t pattern, uint32_t type, std::ostream& ofile) {
	ofile << "Vector " << pattern << ":\t";
	for (uint32_t i = 0; i < ckt.size(); i++) {
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
						ofile << std::setw(OUTJUST) << (uint32_t)in[i] << " "; break;
				} break;
			case 3: 
				switch ((uint32_t)in[i]) {
					case 0: ofile << std::setw(OUTJUST) << "N" << " "; break;
					case 1: ofile << std::setw(OUTJUST) << "Y" << " "; break;
					default: ofile << std::setw(OUTJUST) << (uint32_t)in[i] << " "; break;
				} break;
			default:
				if (ckt.at(i).typ == INPT)
					ofile << std::setw(OUTJUST) << (uint32_t)in[i] << " "; break;
		}
	}
	ofile << std::endl;
}

float serial(Circuit& ckt, CPU_Data& input, uint64_t** covered) {
	std::ofstream s1file("serialsim-p1.log", std::ios::out);
	std::ofstream s2file("serialsim-p2.log", std::ios::out);
	std::ofstream mfile("serialmark.log", std::ios::out);
	std::ofstream cfile("serialcover.log", std::ios::out);
    float total = 0.0, elapsed;
    timespec start, stop;
    uint32_t* simulate;
    uint32_t* mark; 
    uint32_t* merge = new uint32_t[ckt.size()];
    int32_t* mergeLog = new int32_t[ckt.size()];
    uint32_t* cover = new uint32_t[ckt.size()];
    uint32_t* hist_cover;
	*covered = new uint64_t;
	uint64_t* coverage = *covered;
    *coverage = 0;
	
/*	std::cerr << "CPU results:" << std::endl;
	std::clog << "Line:   \t";
	for (uint32_t i = 0; i < ckt.size(); i++) { 
		DPRINT("%3d ", i);
	}
	std::clog << std::endl;
*/
	for (uint32_t i = 0; i < ckt.size(); i++) {
		merge[i] = 0;
		mergeLog[i] = -1;
	}
    for (uint32_t pattern = 0; pattern < input.width(); pattern++) {
        simulate = new uint32_t[ckt.size()];
        mark = new uint32_t[ckt.size()];
		cover = new uint32_t[ckt.size()];
		hist_cover = new uint32_t[ckt.size()];
        for (uint32_t i = 0; i < ckt.size(); i++) {
            simulate[i] = 0;
            mark[i] = 0;
			cover[i] = 0;
			hist_cover[i] = 0;
        }
		try { 
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
			// simulate pattern 1
			//        std::cerr << "Serial Simulate P1" << std::endl;
			cpuSimulateP1(ckt, input.cpu().data, simulate, input.cpu().pitch,pattern);
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
			elapsed = floattime(diff(start, stop));
			total += elapsed;
		} catch(std::exception e) { 
			std::cerr << "Caught exception in cpuSim Pass 1" << e.what() << std::endl;
		}
		debugPrintSim(ckt, simulate,pattern, 2, s1file);
		// simulate pattern 2
		try { 
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
			if (pattern == (input.width()-1))  {
				cpuSimulateP2(ckt, input.cpu().data, simulate, input.cpu().pitch,0);
			}
			else {
				cpuSimulateP2(ckt, input.cpu().data, simulate, input.cpu().pitch,pattern+1);
			}
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
			elapsed = floattime(diff(start, stop));
			total += elapsed;
		} catch(std::exception e) { 
			std::cerr << "Caught exception in cpuSim Pass 2" << e.what() << std::endl;
		}
		debugPrintSim(ckt, simulate,pattern, 2, s2file);
        // mark
		try {
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
			cpuMark(ckt, simulate, mark);
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
			elapsed = floattime(diff(start, stop));
			total += elapsed;
		} catch(std::exception e) { 
			std::cerr << "Caught exception in cpuMark" << e.what() << std::endl;
		}
		//std::cerr << "    Mark: ";
		debugPrintSim(ckt, mark,pattern, 3, mfile);
        // calculate coverage against all previous runs
		try { 
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	        cpuCover(ckt, mark, merge, hist_cover, cover, coverage);
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
			elapsed = floattime(diff(start, stop));
			total += elapsed;
		} catch (std::exception e) {
			std::cerr << "Caught an exception in cpuCover - " << e.what() << std::endl;
		}
		debugPrintSim(ckt, cover,pattern, 4, cfile);
        // merge mark to history
        //std::cerr << "Merge" << std::endl;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        cpuMerge(ckt, mark, merge);
		cpuMergeLog(ckt, mark, mergeLog, pattern);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        elapsed = floattime(diff(start, stop));
//		std::cerr << "Pass " << pattern << " time: " << elapsed<<std::endl;
        total += elapsed;
        delete mark;
        delete simulate;
		delete cover;
		delete hist_cover;
    }
	debugMergeOutput(mergeLog, 1, ckt.size(), "serialmerge.log" );
    DPRINT("Serial Coverage: %lu\n", *coverage);
    delete coverage;
	s1file.close();
	s2file.close();
	mfile.close();
	cfile.close();
	std::clog << "Completed serial simulation." << std::endl;
    return total;
}

void debugMergeOutput(int32_t* data, size_t height, size_t width, std::string outfile) {
#ifndef NDEBUG
	std::ofstream ofile(outfile.c_str());
	ofile << "Size: " << width << "x" << height << " WxH " << std::endl;
	for (uint32_t r = 0;r < width; r++) {
		ofile << "Gate " << r << ":\t";
		for (uint32_t i = 0; i < height; i++) {
			int32_t z = data[r];
			switch(z) {
				default:
					ofile << std::setw(OUTJUST) << (int32_t)z << " "; break;
			}
		}
		ofile << std::endl;
	}
	ofile.close();
#endif

}
uint32_t gateeval (uint32_t f1, uint32_t f2, uint32_t type) {
	uint32_t nand2[16] = {1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0};
	uint32_t and2[16]  = {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1};
	uint32_t nor2[16]  = {1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0};
	uint32_t or2[16]   = {0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
	uint32_t xnor2[16] = {1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1};
	uint32_t xor2[16]  = {0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0};
	uint32_t val = 0xff;
	switch(type) {
		case AND: 
			val = REF2D(uint32_t, and2, sizeof(uint32_t)*4, f1, f2); break;
		case NAND:
			val = REF2D(uint32_t, nand2, sizeof(uint32_t)*4, f1, f2); break;
		case OR:
			val = REF2D(uint32_t, or2, sizeof(uint32_t)*4, f1, f2); break;
		case NOR: 
			val = REF2D(uint32_t, nor2, sizeof(uint32_t)*4, f1, f2); break;
		case XOR: 
			val = REF2D(uint32_t, xor2, sizeof(uint32_t)*4, f1, f2); break;
		case XNOR:
			val = REF2D(uint32_t, xnor2, sizeof(uint32_t)*4, f1, f2); break;
	}
	return val;
}

void cpuSimulateP1(const Circuit& ckt, uint8_t* pi, uint32_t* sim, size_t pi_pitch, size_t pattern) {
    for (uint32_t g = 0; g < ckt.size(); g++) {
        const NODEC gate = ckt.at(g);
        uint32_t val=0;
        switch(gate.typ) {
            case INPT:
                val = REF2D(uint8_t,pi,pi_pitch,pattern,g); break;
			case FROM:
				val = FREF(sim,gate,fin,0); break;
            default:
                if (gate.typ != NOT) {
                    val = FREF(sim,gate,fin,0);
                } else {
                    val = (FREF(sim,gate,fin,0) != 1);
                }
                uint32_t j = 1;
                while (j < gate.fin.size()) {
					val = gateeval(val,sim[gate.fin.at(j).second],gate.typ);
					j++;
                }
        }
        sim[g] = val;
    }
}
void cpuSimulateP2(const Circuit& ckt, uint8_t* pi, uint32_t* sim,size_t pi_pitch, size_t pattern) {
	uint32_t stable[2][2] = {{S0, T1}, {T0, S1}};
    for (uint32_t g = 0; g < ckt.size(); g++) {
        const NODEC gate = ckt.at(g);
        uint32_t val;
        switch(gate.typ) {
            case INPT: val = REF2D(uint8_t,pi,pi_pitch,pattern, g); break;
			case FROM: val = BIN(FREF(sim,gate,fin,0)); break;
            default:
                if (gate.typ != NOT) {
                    val = BIN(FREF(sim,gate,fin,0));
                } else {
                    val = (BIN(FREF(sim,gate,fin,0)) != 1);
                }
              uint32_t j = 1;
              while (j < gate.fin.size()) {
					val = gateeval(val,FREF(sim,gate,fin,j),gate.typ);
					j++;
                }
 
        }
		sim[g] = stable[sim[g]][val];
    }

}
uint32_t cpuMarkEval_in(uint32_t f1, uint32_t f2, uint32_t type) {
	uint32_t and2_input_prop[16] = {0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1};
	uint32_t or2_input_prop[16]  = {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	uint32_t xor2_input_prop[16] = {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	switch(type){
		case AND:
		case NAND:
			return REF2D(uint32_t,and2_input_prop,sizeof(uint32_t)*4,f1,f2);
		case NOR:
		case OR:
			return REF2D(uint32_t,or2_input_prop,sizeof(uint32_t)*4,f1,f2);
		case XOR:
		case XNOR:
			return REF2D(uint32_t,xor2_input_prop,sizeof(uint32_t)*4,f1,f2);
	}
	return 0xff;
}
uint32_t cpuMarkEval_out(uint32_t f1, uint32_t f2, uint32_t type) {
	uint32_t and2_output_prop[16]= {0,0,0,0,0,2,1,1,0,1,1,0,0,1,0,1};
	uint32_t or2_output_prop[16] = {2,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1};
	uint32_t xor2_output_prop[16]= {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	switch(type){
		case AND:
		case NAND:
			return REF2D(uint32_t,and2_output_prop,sizeof(uint32_t)*4,f1,f2);
		case NOR:
		case OR:
			return REF2D(uint32_t,or2_output_prop,sizeof(uint32_t)*4,f1,f2);
		case XOR:
		case XNOR:
			return REF2D(uint32_t,xor2_output_prop,sizeof(uint32_t)*4,f1,f2);
	}
	return 0xff;
}
void cpuMark(const Circuit& ckt, uint32_t* sim, uint32_t* mark) {
    uint32_t fin;
	uint8_t resultCache, cache, rowCache, val, prev, pass, tmp = 1;
	uint32_t fin1 = 0, fin2 = 0;
	for (uint32_t g2 = 0; g2 < ckt.size(); g2++) {
		uint32_t g = (ckt.size() - (g2+1));
		try {
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
				for (uint32_t i = 0; i < gate.nfo; i++) {
					resultCache = (resultCache ==1) || (FREF(mark,gate,fot,i) > 0);
				}
				prev = resultCache;
			}
			switch(gate.typ) {
				case INPT:
					if (gate.nfo == 0 && gate.nfi == 0) {
						resultCache = 0; // on the odd case that an input is literally connected to nothing, this is not a path.
					};
				break;
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
				case OR:
				case NOR:
				case XOR:
				case XNOR:
				case NAND:
				case AND:
						   for (fin1 = 0; fin1 < gate.nfi; fin1++) {
							   fin = 1;
							   for (fin2 = 0; fin2 < gate.nfi; fin2++) {
								   if (fin1 != fin2) {
									   cache = cpuMarkEval_out(FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2),gate.typ);
									   pass += (cache > 1);
									   tmp = tmp && (cache > 0);
									   if (gate.nfi > 1) {
										   cache = cpuMarkEval_in(FREF(sim,gate,fin,fin1), FREF(sim,gate,fin,fin2),gate.typ);
										   fin = cache && fin && prev;
									   }
								   }
							   }
							   FREF(mark,gate,fin,fin1) = fin;
						   }
						   break;
				default:
						   // if there is a transition that will propagate, set = to some positive #?
						   break;
			}

		} catch (std::out_of_range e) {
			std::cerr << "Failed to get node. " << g << "/" << ckt.size() << std::endl;
			throw e;
		}
		// stick the contents of resultCache uint32_to the results array
		mark[g] = resultCache;
	}
}
void cpuCover(const Circuit& ckt, uint32_t* mark, uint32_t* hist, uint32_t* hist_cover, uint32_t* cover, uint64_t* covered) {
    // cover is the coverage uint32_ts we're working with for this pass.
    // mark is the fresh marks
    // hist is the history of the mark status of all lines.
    for (uint32_t g2 = 0; g2 < ckt.size(); g2--) {
		uint32_t g = ckt.size() - (g2+1);
        const NODEC& gate = ckt.at(g);
        if (gate.po == true) {
            cover[g] = 0;
            hist_cover[g] = (mark[g] > 0);
        }
		if (gate.nfo > 1) {
			for (uint32_t i = 0; i < gate.nfo; i++) {
				cover[g] += FREF(cover,gate,fot,i);
				hist_cover[g] += FREF(hist_cover,gate,fot,i);
			}
		}
		if (gate.typ != FROM) {
			cover[g] = (NOTMARKED(mark,hist,g))*(cover[g]+hist_cover[g]);
			hist_cover[g] = (NOTMARKED(mark,hist,g) ? 0 : hist_cover[g]);
            for (uint32_t i = 0; i < gate.nfi; i++) {
				FREF(cover,gate,fin,i) = cover[g];
				FREF(hist_cover,gate,fin,i) = hist_cover[g];
            }
        } 
		if (gate.typ == INPT)
            *covered = *covered + cover[g];
		//hist[g] |= mark[g];
	}
}
