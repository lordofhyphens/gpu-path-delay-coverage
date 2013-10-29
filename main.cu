
static int cover_flag = 0;
static int sim_flag = 0;
static int merge_flag = 0;
static int mark_flag = 0;
static int verbose_flag = 0;
static int robust_flag = 0;

#include "util/utility.h"
#include "util/ckt.h"
#include "util/gpuckt.h"
#include "util/gpudata.h"
#include "util/vectors.h"
#include "simkernel.cuh"
#include "markkernel.cuh"
#include "mergekernel.cuh"
#include "coverkernel.cuh"
#include "util/subckt.h"
#include <utility>
#include <iostream>
#include <fstream>
#include <getopt.h>
#define MAX_PATTERNS simul_patterns
using namespace std;
#undef OUTJUST
#define OUTJUST 4


// default value for segments, can be overriden on command line
static int segs = 1; 

int main(int argc, char* argv[]) {
	int c;
	uint8_t device = selectGPU();
	extern int optind;
	resetGPU(); // ensures that we have a new GPU context to work with.
	GPU_Circuit ckt;
	timespec start;
	float mark=0.0,merge =0.0,cover = 0.0,sim1 = 0.0,gpu =0.0;
	std::string infile;
	std::vector<string> benchmarks;
	int option_index = 0;
	int override_patterns = 0;

	{ // Loop to process incoming command line arguments.
	while (1) {
		static struct option long_options[] =
		{
			/* These options set a flag. */
			{"verbose", no_argument,       &verbose_flag, 1},
			{"cover", no_argument,       &cover_flag, 1},
			{"mark", no_argument,       &mark_flag, 1},
			{"merge", no_argument,       &merge_flag, 1},
			{"sim", no_argument,       &sim_flag, 1},
			{"robust", no_argument,       &robust_flag, 1},
			{"brief",   no_argument,       &verbose_flag, 0},
			/* These options don't set a flag.
			   We distinguish them by their indices. */
			{"help",     no_argument,       0, 'h'},
			{"bench",     required_argument,       0, 'b'},
			{"num_patterns",     required_argument,       0, 'o'},
			{"segs",    required_argument, 0, 's'},
			{0, 0}
		};
		/* getopt_long stores the option index here. */
		c = getopt_long (argc, argv, "b:s:",
				long_options, &option_index);

		/* Detect the end of the options. */
		if (c == -1)
			break;

		switch (c)
		{
			case 0:
				/* If this option set a flag, do nothing else now. */
				if (long_options[option_index].flag != 0)
					break;
				printf ("option %s", long_options[option_index].name);
				if (optarg)
					printf (" with arg %s", optarg);
				printf ("\n");
	 			break;

			case 'b':
				infile = std::string(optarg);
				break;

			case 't':
				printf ("option %s = %s\n", long_options[option_index].name, optarg);
				benchmarks.push_back(std::string(optarg));
				break;
			case 'o':
				override_patterns  = atoi(optarg);
				break;

			case 's':
				segs = atoi(optarg);
				if (segs > 8) {
					std::cerr << "Error: Segment lengths over 8 are not supported!\n";
					abort();
				}
				break;
			case 'h':
				printf("Usage: %s (options) /path/to/benchmark1 /path/to/benchmark2\n", argv[0]);
				printf("\t--segs N : Run fault grading over segments of length N, where 1 <= N <= 8\n");
				printf("\t--bench /path/to/ckt : A circuit to apply benchmarks.\n");
				abort();
			case '?':
				/* getopt_long already printed an error message. */
				break;

			default:
				abort ();
		}
	}
	if (optind < argc) {
		/* these are the arguments after the command-line options */ 
		for (; optind < argc; optind++) 
				benchmarks.push_back(std::string(argv[optind]));
	} else { 
		printf("no arguments left to process\n"); 
	} }
	// done with getopts, rest are benchmark vectors.
	if (infile.empty()) {
		printf("--bench argument is required.");
		abort();
	}
	if (infile.find("bench") != std::string::npos) {
		ckt.read_bench(infile.c_str());
	} else {
			std::clog << "presorted benchmark " << infile << " ";
		ckt.load(infile.c_str());
	}

	std::cerr << "Copying circuit to GPU...";
	ckt.copy(); // convert and copy circuit to GPU
	std::clog << "Circuit size is: " << ckt.size() << "Levels: " << ckt.levels() << std::endl;

	for (uint32_t i = 0; i < benchmarks.size(); i++) { // run multiple benchmark values from the same program invocation
		uint64_t *totals = new uint64_t; 
		std::string vector_file = benchmarks[i];
		*totals = 0;
		gpu = 0.0;
		std::cerr << "Vector set " << vector_file << std::endl;
		std::pair<size_t,size_t> vecdim = get_vector_dim(vector_file.c_str());
		assert(vecdim.first > 0);
		std::cerr << "Vector size: " << vecdim.first << "x"<<vecdim.second << std::endl;
		GPU_Data *vec = new GPU_Data(vecdim.first,vecdim.second, vecdim.first);
		uint32_t simul_patterns = gpuCalculateSimulPatterns(ckt.size(), vecdim.first, device);
		if (override_patterns > 0) {
			simul_patterns = override_patterns-1;
		}

		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		read_vectors(*vec, vector_file.c_str(), vec->block_width(), vecdim.first);
		debugDataOutput(*vec, "vecout.log");

		std::clog << "Maximum patterns per pass: " << simul_patterns << " / " << vecdim.first << std::endl;

		std::cerr << "Initializing gpu memory for results...";
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		GPU_Data *sim_results = new GPU_Data(vecdim.first,ckt.size(), MAX_PATTERNS); // initializing results array for simulation
		std::clog << "Maximum patterns per pass: " << simul_patterns << "("<<sim_results->block_width() << ")" << " / " << vecdim.first << std::endl;
		gpu += elapsed(start);

		std::cerr << "..complete." << std::endl;
		size_t startPattern = 0;
		void *dc_segs = NULL;
		int segments = 0;
		for (unsigned int chunk = 0; chunk < sim_results->size(); chunk++) {
			uint64_t *coverage = new uint64_t; 
			*coverage = 0;
			std::clog << "Simulation ...";
			sim1 = gpuRunSimulation(*sim_results, *vec, ckt, chunk, startPattern, chunk+1 == sim_results->size());
			std::clog << "..complete." << std::endl;
			gpu += sim1;
			std::cerr << "Simulation: " << sim1 << " ms" << std::endl;
			// don't need the input vectors anymore, so remove.
			GPU_Data *mark_results = new GPU_Data(vecdim.first,ckt.size(), MAX_PATTERNS);
			// quick test of clear code
			mark = gpuMarkPaths(*mark_results, *sim_results, ckt, chunk, startPattern);
			gpu += mark;
			std::cerr << "     Mark: " << mark << " ms" << std::endl;
			switch(segs) {
				case 1:
					merge = gpuMergeSegments<1>(*mark_results, *sim_results, ckt, chunk, startPattern, &dc_segs, segments); break;
				case 2:
					merge = gpuMergeSegments<2>(*mark_results, *sim_results, ckt, chunk, startPattern, &dc_segs, segments); break;
				case 3:
					merge = gpuMergeSegments<3>(*mark_results, *sim_results, ckt, chunk, startPattern, &dc_segs, segments); break;
				case 4:
					merge = gpuMergeSegments<4>(*mark_results, *sim_results, ckt, chunk, startPattern, &dc_segs, segments); break;
				case 5:
					merge = gpuMergeSegments<5>(*mark_results, *sim_results, ckt, chunk, startPattern, &dc_segs, segments); break;
				case 6:
					merge = gpuMergeSegments<6>(*mark_results, *sim_results, ckt, chunk, startPattern, &dc_segs, segments); break;
				case 7:
					merge = gpuMergeSegments<7>(*mark_results, *sim_results, ckt, chunk, startPattern, &dc_segs, segments); break;
				case 8:
					merge = gpuMergeSegments<8>(*mark_results, *sim_results, ckt, chunk, startPattern, &dc_segs, segments); break;
			}
			gpu += merge;
			std::cerr << " Merge: " << merge << " ms" << std::endl;
			sim_results->unload();
			switch(segs) {
				case 1:
					cover = gpuCountPaths<1>(ckt, *mark_results, dc_segs, segments, coverage, chunk, startPattern); break;
				case 2:
					cover = gpuCountPaths<2>(ckt, *mark_results, dc_segs, segments, coverage, chunk, startPattern); break;
				case 3: 
					cover = gpuCountPaths<3>(ckt, *mark_results, dc_segs, segments, coverage, chunk, startPattern); break;
				case 4:
					cover = gpuCountPaths<4>(ckt, *mark_results, dc_segs, segments, coverage, chunk, startPattern); break;
				case 5:
					cover = gpuCountPaths<5>(ckt, *mark_results, dc_segs, segments, coverage, chunk, startPattern); break;
				case 6:
					cover = gpuCountPaths<6>(ckt, *mark_results, dc_segs, segments, coverage, chunk, startPattern); break;
				case 7:
					cover = gpuCountPaths<7>(ckt, *mark_results, dc_segs, segments, coverage, chunk, startPattern); break;
				case 8:
					cover = gpuCountPaths<8>(ckt, *mark_results, dc_segs, segments, coverage, chunk, startPattern); break;
			}
			*totals += *coverage;
			std::cerr << " Cover: " << cover << " ms" << std::endl;
			std::cerr << "GPU Coverage: " << *coverage << ", total: "<< *totals << std::endl;
			gpu += cover;
			startPattern += mark_results->block_width();
			mark_results->unload();
			delete coverage;
		}
		std::cerr << "   GPU: " << gpu << " ms" <<std::endl;
		std::cout << vector_file << ":" << vecdim.first << "," << ckt.size() << "," << segs <<  ";" << gpu << ";" << sim1 
			      <<  ";" << mark << ";"<< merge << ";" << cover << ";" << *totals << std::endl;
	}
	return 0;
}
