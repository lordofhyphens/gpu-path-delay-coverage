#include "markkernel.h"
#include <cuda.h>
void HandleMarkError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleMarkError( err, __FILE__, __LINE__ ))
texture<uint8_t, 2> and2OutputPropLUT;
texture<uint8_t, 2> and2InputPropLUT;
texture<uint8_t, 2> or2OutputPropLUT;
texture<uint8_t, 2> or2InputPropLUT;
texture<uint8_t, 2> xor2OutputPropLUT;
texture<uint8_t, 2> xor2InputPropLUT;
texture<uint8_t, 2> fromPropLUT;
texture<uint8_t, 2> inptPropLUT;
texture<uint8_t, 2> mergeLUT;

texture<uint8_t, 2> AndInChainLUT;
texture<uint8_t, 2> AndOutChainLUT;
texture<uint8_t, 2> OrInChainLUT;
texture<uint8_t, 2> OrOutChainLUT;
texture<uint8_t, 2> XorInChainLUT;
texture<uint8_t, 2> XorOutChainLUT;
texture<uint8_t, 2> inputTexture;


void loadPropLUTs() {
	// Creating a set of static arrays that represent our LUTs
		// Addressing for the propagations:
	// 2 4x4 groups such that 
	uint8_t and2_output_prop[16]= {0,0,0,0,0,2,1,1,0,1,1,0,0,1,0,1};
	uint8_t and2_input_prop[16] = {0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1};
	uint8_t or2_output_prop[16] = {2,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1};
	uint8_t or2_input_prop[16]  = {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	uint8_t xor2_input_prop[16] = {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	uint8_t xor2_output_prop[16]= {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	uint8_t from_prop[16]       = {0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1};
	uint8_t inpt_prop[8]        = {0,0,0,0,0,0,1,1};

	uint8_t and_outp_chain[8]   = {0,0,1,0,0,1,1,1};
	uint8_t and_inp_chain[8]    = {0,0,0,0,0,1,1,1};
	uint8_t or_outp_chain[8]    = {0,0,0,1,1,0,1,1};
	uint8_t or_inp_chain[8]     = {0,0,0,0,1,0,1,1};
	uint8_t xor_outp_chain[8]   = {0,0,0,0,0,0,0,0};
	uint8_t xor_inp_chain[8]    = {0,0,0,0,0,0,0,0};

	cudaExtent volumeSize = make_cudaExtent(4,4,2);
	// device memory arrays, required. 
	cudaArray *cuAndInptProp, *cuAndOutpProp, *cuOrInptProp, *cuOrOutpProp, *cuFromProp, *cuInptProp, *cuXorInptProp, *cuXorOutpProp;
	cudaArray *cuAndOutChain, *cuAndInChain, *cuOrInChain, *cuOrOutChain, *cuXorInChain, *cuXorOutChain;
	// generic formatting information. All of our arrays are the same, so sharing it shouldn't be a problem.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();
	// Allocating memory on the device.
		
	HANDLE_ERROR(cudaMallocArray(&cuFromProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuInptProp, &channelDesc, 4,2));
	HANDLE_ERROR(cudaMallocArray(&cuAndInptProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuAndOutpProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuOrInptProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuOrOutpProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuXorInptProp, &channelDesc, 4,4));
	HANDLE_ERROR(cudaMallocArray(&cuXorOutpProp, &channelDesc, 4,4));

	HANDLE_ERROR(cudaMallocArray(&cuXorOutChain, &channelDesc, 4,2));
	HANDLE_ERROR(cudaMallocArray(&cuXorInChain, &channelDesc, 4,2));
	HANDLE_ERROR(cudaMallocArray(&cuOrInChain, &channelDesc, 4,2));
	HANDLE_ERROR(cudaMallocArray(&cuOrOutChain, &channelDesc, 4,2));
	HANDLE_ERROR(cudaMallocArray(&cuAndInChain, &channelDesc, 4,2));
	HANDLE_ERROR(cudaMallocArray(&cuAndOutChain, &channelDesc, 4,2));

	// Copying the LUTs Host->Device
	HANDLE_ERROR(cudaMemcpyToArray(cuFromProp, 0,0, from_prop, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndInptProp, 0,0, and2_input_prop, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndOutpProp, 0,0, and2_output_prop, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrInptProp, 0,0, or2_input_prop, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrOutpProp, 0,0, or2_output_prop, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorInptProp, 0,0, xor2_input_prop, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorOutpProp, 0,0, xor2_output_prop, sizeof(uint8_t)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuInptProp, 0,0, inpt_prop, sizeof(uint8_t)*8,cudaMemcpyHostToDevice));
	
	HANDLE_ERROR(cudaMemcpyToArray(cuXorInChain, 0,0, xor_inp_chain, sizeof(uint8_t)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorOutChain, 0,0, xor_outp_chain, sizeof(uint8_t)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrInChain, 0,0, or_inp_chain, sizeof(uint8_t)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrOutChain, 0,0, or_outp_chain, sizeof(uint8_t)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndInChain, 0,0, and_inp_chain, sizeof(uint8_t)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndOutChain, 0,0, and_outp_chain, sizeof(uint8_t)*8,cudaMemcpyHostToDevice));

	// Marking them as textures. LUTs should be in texture memory and cached on
	// access.
	HANDLE_ERROR(cudaBindTextureToArray(and2OutputPropLUT,cuAndOutpProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(and2InputPropLUT,cuAndInptProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(or2OutputPropLUT,cuOrOutpProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(or2InputPropLUT,cuOrInptProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xor2InputPropLUT,cuXorInptProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(xor2OutputPropLUT,cuXorOutpProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(fromPropLUT,cuFromProp,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(inptPropLUT,cuInptProp,channelDesc));
	
	HANDLE_ERROR(cudaBindTextureToArray(XorOutChainLUT,cuXorOutChain,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(XorInChainLUT,cuXorInChain,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(OrOutChainLUT,cuOrOutChain,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(OrInChainLUT,cuOrInChain,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(AndOutChainLUT,cuAndOutChain,channelDesc));
	HANDLE_ERROR(cudaBindTextureToArray(AndInChainLUT,cuAndInChain,channelDesc));
}
__device__ uint8_t markeval_out (uint8_t f1, uint8_t f2, int type) {

	switch(type) {
		case AND:
		case NAND:
			return tex2D(and2OutputPropLUT, f1, f2);
		case OR:
		case NOR:
			return tex2D(or2OutputPropLUT, f1, f2);
		case XOR:
		case XNOR:
			return tex2D(xor2OutputPropLUT, f1, f2);
	}
	return 0xff;
}
__device__ uint8_t markeval_in (uint8_t f1, uint8_t f2, int type) {
	switch(type) {
		case AND:
		case NAND:
			return tex2D(and2InputPropLUT, f1, f2);
		case OR:
		case NOR:
			return tex2D(or2InputPropLUT, f1, f2);
		case XOR:
		case XNOR:
			return tex2D(xor2InputPropLUT, f1, f2);
	}
	return 0xff;
}

__global__ void kernMarkPathSegments(uint8_t *sim, size_t sim_pitch, uint8_t* mark, size_t pitch, size_t patterns, GPUNODE* node, uint32_t* fans, int start, int startPattern) {
	int tid = (blockIdx.y * blockDim.x) + threadIdx.x, nfi, goffset,val,prev;
	int gid = (blockIdx.x) + start;
	uint8_t rowCache, resultCache;
	uint8_t cache, fin = 1;
	int tmp = 1, pass = 0, fin1 = 0, fin2 = 0,type;
	if (tid < patterns) {
		cache = 0;
		rowCache = REF2D(uint8_t,sim,sim_pitch,tid,gid);
		resultCache = REF2D(uint8_t,mark,pitch,tid,gid);
		tmp = 1;
		nfi = node[gid].nfi;
		type = node[gid].type;
		goffset = node[gid].offset;
		__syncthreads();
		// switching based on value causes divergence, switch based on node type.
		// rowCache is from the simulation mark. 0-1, stable, 2-3, transition
		val = (rowCache > 1);

		if (node[gid].po > 0) {
			resultCache = val;
			prev = val;
		} else {
			prev = resultCache;
		}
		if (node[gid].nfo > 1) {
			prev = 0;
			resultCache = 0;
			for (int i = 0; i < node[gid].nfo; i++) {
				resultCache = (resultCache == 1) || (REF2D(uint8_t,mark,pitch,tid,FIN(fans,goffset,i+node[gid].nfi)) > 0);
			}
			prev = resultCache;
		}
		switch(type) {
			case INPT:
				if (node[gid].nfo == 0 && node[gid].nfi == 0) {
					resultCache = 0; // on the odd case that an input is literally connected to nothing, this is not a path.
				}
				break;
			case FROM: break;
			case BUFF:
			case NOT:
				val = NOT_IN(rowCache) && prev;
				REF2D(uint8_t,mark,pitch,tid,FIN(fans,goffset,0)) = val;
				resultCache = val;
				break;
				// For the standard gates, setting three values -- both the
				// sim lines and the output line.  rowCache[threadIdx.x][i]-1 is the
				// transition on the output, offset to make the texture
				// calculations correct because there are 4 possible values
				// rowCache[threadIdx.x][i] can take: 0, 1, 2, 3.  0, 1 are the same, as are
				// 2,3, so we subtract 1 and clamp to an edge if we
				// overflow.
				// 0 becomes -1 (which clamps to 0)
				// 1 becomes 0
				// 2 becomes 1
				// 3 becomes 2 (which clamps to 1)
				// There's only 2 LUTs for each gate type. The sim LUT
				// checks for path existance through the first sim, so we
				// call it twice with the sims reversed to check both
				// paths.
			case OR:
			case NOR:
			case XOR:
			case XNOR:
			case NAND:
			case AND:
				for (fin1 = 0; fin1 < node[gid].nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < nfi; fin2++) {
						if (fin1 != fin2) {
							cache = markeval_out(REF2D(uint8_t,sim,sim_pitch,tid,FIN(fans,goffset,fin1)),REF2D(uint8_t,sim,sim_pitch,tid,FIN(fans,goffset,fin2)), type);
							pass += (cache > 1);
							tmp = tmp && (cache > 0);
							if (nfi > 1) {
								cache = markeval_in(REF2D(uint8_t,sim,sim_pitch,tid,FIN(fans,goffset,fin1)),REF2D(uint8_t,sim,sim_pitch,tid,FIN(fans,goffset,fin2)), type);
								fin = cache && fin && prev;
							}
						}
					}
					REF2D(uint8_t,mark,pitch,tid,FIN(fans,goffset,fin1)) = fin;
				}
				break;
			default: break;
		}
		// stick the contents of resultCache into the mark array
		REF2D(uint8_t,mark,pitch,tid,gid) = resultCache;


	}
}

float gpuMarkPaths(GPU_Data& results, GPU_Data& input, GPU_Circuit& ckt) {
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting before we hit this point.
	loadPropLUTs();
	int startGate;
	int blockcount_y;
#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	int startPattern = 0;
	for (unsigned int chunk = 0; chunk < input.size(); chunk++) {
		blockcount_y = (int)(results.gpu(chunk).width/MARK_BLOCK) + ((results.gpu(chunk).width% MARK_BLOCK) > 0);
		startGate=ckt.size()-1;
		for (int i = ckt.levels(); i >= 0; i--) {
			int levelsize = ckt.levelsize(i);
			do { 
				int simblocks = min(MAX_BLOCKS, levelsize);
				dim3 numBlocks(simblocks,blockcount_y);
				startGate -= simblocks;
				kernMarkPathSegments<<<numBlocks,MARK_BLOCK>>>(input.gpu(chunk).data, input.gpu(chunk).pitch, results.gpu(chunk).data, results.gpu(chunk).pitch, results.gpu(chunk).width,ckt.gpu_graph(), ckt.offset(),  startGate+1, startPattern);
				if (levelsize > MAX_BLOCKS) {
					levelsize -= MAX_BLOCKS;
				} else {
					levelsize = 0;
				}
				cudaDeviceSynchronize();
			} while (levelsize > 0);
			HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
		}
		startPattern += input.gpu(chunk).width;
	}
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}

void debugMarkOutput(GPU_Data* results, std::string outfile = "simdebug.log") {
#ifndef NDEBUG
	std::ofstream ofile(outfile.c_str());
	size_t t = 0;
	for (size_t chunk = 0; chunk < results->size(); chunk++) {
		uint8_t *lvalues;
		lvalues = (uint8_t*)malloc(results->gpu(chunk).height*results->gpu(chunk).pitch);
		cudaMemcpy2D(lvalues,results->gpu().pitch,results->gpu(chunk).data,results->gpu(chunk).pitch,results->gpu(chunk).width,results->gpu(chunk).height,cudaMemcpyDeviceToHost);
		for (unsigned int r = 0;r < results->gpu(chunk).width; r++) {
			ofile << "Vector " << t << ":\t";
			for (unsigned int i = 0; i < results->gpu(chunk).height; i++) {
				uint8_t z = REF2D(uint8_t, lvalues, results->gpu(chunk).pitch, r, i);
				switch(z) {
					case 0:
						ofile  << std::setw(OUTJUST) << "N" << " "; break;
					case 1:
						ofile  << std::setw(OUTJUST) << "Y" << " "; break;
					default:
						ofile << std::setw(OUTJUST) << (int)z << " "; break;
				}
			}
			ofile << std::endl;
			t++;
		}
		free(lvalues);
	}
	ofile.close();
#endif

}
void debugMarkOutput(ARRAY2D<uint8_t> results, std::string outfile) {
#ifndef NDEBUG
	uint8_t *lvalues;
	std::ofstream ofile(outfile.c_str());
//	ofile << "Line:   \t";
//	for (unsigned int i = 0; i < results.height; i++) {
//		ofile << std::setw(OUTJUST) << i << " ";
//	}
//	ofile << std::endl;
	lvalues = (uint8_t*)malloc(results.height*results.pitch);
	cudaMemcpy2D(lvalues,results.pitch,results.data,results.pitch,results.width,results.height,cudaMemcpyDeviceToHost);
	for (unsigned int r = 0;r < results.width; r++) {
		ofile << "Vector " << r << ":\t";
		for (unsigned int i = 0; i < results.height; i++) {
			uint8_t z = REF2D(uint8_t, lvalues, results.pitch, r, i);
			switch(z) {
				case 0:
					ofile  << std::setw(OUTJUST) << "N" << " "; break;
				case 1:
					ofile  << std::setw(OUTJUST) << "Y" << " "; break;
				default:
					ofile << std::setw(OUTJUST) << (int)z << " "; break;
			}
		}
		ofile << std::endl;
	}
	free(lvalues);
	ofile.close();
#endif
}

