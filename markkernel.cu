#include "markkernel.h"
#include <cuda.h>
void HandleMarkError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        DPRINT( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleMarkError( err, __FILE__, __LINE__ ))
texture<int, 2> and2OutputPropLUT;
texture<int, 2> and2InputPropLUT;
texture<int, 2> or2OutputPropLUT;
texture<int, 2> or2InputPropLUT;
texture<int, 2> xor2OutputPropLUT;
texture<int, 2> xor2InputPropLUT;
texture<int, 2> fromPropLUT;
texture<int, 2> inptPropLUT;
texture<int, 2> mergeLUT;

texture<int, 2> AndInChainLUT;
texture<int, 2> AndOutChainLUT;
texture<int, 2> OrInChainLUT;
texture<int, 2> OrOutChainLUT;
texture<int, 2> XorInChainLUT;
texture<int, 2> XorOutChainLUT;
texture<char, 2> inputTexture;


void loadPropLUTs() {
	// Creating a set of static arrays that represent our LUTs
		// Addressing for the propagations:
	// 2 4x4 groups such that 
	int and2_output_prop[16]= {0,0,0,0,0,2,1,1,0,1,1,0,0,1,0,1};
	int and2_input_prop[16] = {0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1};
	int or2_output_prop[16] = {2,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1};
	int or2_input_prop[16]  = {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	int xor2_input_prop[16] = {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	int xor2_output_prop[16]= {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1};
	int from_prop[16]       = {0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1};
	int inpt_prop[8]        = {0,0,0,0,0,0,1,1};

	int and_outp_chain[8]   = {0,0,1,0,0,1,1,1};
	int and_inp_chain[8]    = {0,0,0,0,0,1,1,1};
	int or_outp_chain[8]    = {0,0,0,1,1,0,1,1};
	int or_inp_chain[8]     = {0,0,0,0,1,0,1,1};
	int xor_outp_chain[8]   = {0,0,0,0,0,0,0,0};
	int xor_inp_chain[8]    = {0,0,0,0,0,0,0,0};

	cudaExtent volumeSize = make_cudaExtent(4,4,2);
	// device memory arrays, required. 
	cudaArray *cuAndInptProp, *cuAndOutpProp, *cuOrInptProp, *cuOrOutpProp, *cuFromProp, *cuInptProp, *cuXorInptProp, *cuXorOutpProp;
	cudaArray *cuAndOutChain, *cuAndInChain, *cuOrInChain, *cuOrOutChain, *cuXorInChain, *cuXorOutChain;
	// generic formatting information. All of our arrays are the same, so sharing it shouldn't be a problem.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
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
	HANDLE_ERROR(cudaMemcpyToArray(cuFromProp, 0,0, from_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndInptProp, 0,0, and2_input_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndOutpProp, 0,0, and2_output_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrInptProp, 0,0, or2_input_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrOutpProp, 0,0, or2_output_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorInptProp, 0,0, xor2_input_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorOutpProp, 0,0, xor2_output_prop, sizeof(int)*16,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuInptProp, 0,0, inpt_prop, sizeof(int)*8,cudaMemcpyHostToDevice));
	
	HANDLE_ERROR(cudaMemcpyToArray(cuXorInChain, 0,0, xor_inp_chain, sizeof(int)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuXorOutChain, 0,0, xor_outp_chain, sizeof(int)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrInChain, 0,0, or_inp_chain, sizeof(int)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuOrOutChain, 0,0, or_outp_chain, sizeof(int)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndInChain, 0,0, and_inp_chain, sizeof(int)*8,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToArray(cuAndOutChain, 0,0, and_outp_chain, sizeof(int)*8,cudaMemcpyHostToDevice));

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

__global__ void kernMarkPathSegments(char *input, int inpitch, char* results, GPUNODE* node, int* fans, size_t width, size_t height, int start, int pitch) {
	int tid = (blockIdx.y * blockDim.y) + threadIdx.x, nfi, goffset,val,prev;
	int gid = (blockIdx.x) + start;
	char rowCache, resultCache;
	char cache, fin = 1;
	int tmp = 1, pass = 0, fin1 = 0, fin2 = 0,type;
	char *rowResults;
	char *row;
	if (tid < height) {
//		printf("%s - Line: %d, gate: %d\n",__FILE__, __LINE__,gid);
		cache = 0;
		row = ADDR2D(char,input,inpitch,tid,gid);
		//(char*)((char*)input + gid*pitch);
		rowResults = ADDR2D(char,results,pitch,tid,gid);
		tmp = 1;
		nfi = node[gid].nfi;
		type = node[gid].type;
		goffset = node[gid].offset;
		rowCache = *row;
		resultCache = *rowResults;
		__syncthreads();
		// switching based on value causes divergence, switch based on node type.
		// rowCache is from the simulation results. 0-1, stable, 2-3, transition
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
				resultCache |= REF2D(char,results,pitch,tid,FIN(fans,goffset+node[gid].nfi,i));
			}
			prev = resultCache;
		}
		switch(type) {
			case FROM: break;
			case BUFF:
			case NOT:
				val = tex2D(inptPropLUT, rowCache,resultCache) && prev;
				REF2D(char,results,pitch,tid,FIN(fans,goffset,0)) = val;
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
				for (fin1 = 0; fin1 < nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < nfi; fin2++) {
						if (fin1 == fin2) continue;
						cache = tex2D(and2OutputPropLUT, REF2D(char,input,inpitch,tid,FIN(fans,goffset,fin1)), REF2D(char,input,inpitch,tid,FIN(fans,goffset,fin2)));
						pass += (cache > 1);
						tmp = tmp && (cache > 0);
						if (nfi > 1) {
							cache = tex2D(and2InputPropLUT, REF2D(char,input,inpitch,tid,FIN(fans,goffset,fin1)), REF2D(char,input,inpitch,tid,FIN(fans,goffset,fin2)));
							fin = cache && fin && prev;
						}
					}
					REF2D(char,results,pitch,tid,FIN(fans,goffset,fin1)) = fin;
				}
	//			resultCache= val && tmp && (pass < nfi) && prev;
				break;
			case OR:
			case NOR:
				for (fin1 = 0; fin1 < nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < nfi; fin2++) {
						if (fin1 == fin2) continue;
						cache = tex2D(or2OutputPropLUT, REF2D(char,input,inpitch,tid,FIN(fans,goffset,fin1)), REF2D(char,input,inpitch,tid,FIN(fans,goffset,fin2)));
						pass += (cache > 1);
						tmp = tmp && (cache > 0);

						if (nfi > 1) {
							cache = tex2D(or2InputPropLUT, REF2D(char,input,inpitch,FIN(fans,goffset,fin1),tid), REF2D(char,input,inpitch,FIN(fans,goffset,fin2),tid));
							fin = cache && fin && prev;
						}

					}
					REF2D(char,results,pitch,tid,FIN(fans,goffset,fin1)) = fin;
				}
//				resultCache= val && tmp && (pass <= nfi) && prev;
				break;
			case XOR:
			case XNOR:
				for (fin1 = 0; fin1 < nfi; fin1++) {
					fin = 1;
					for (fin2 = 0; fin2 < nfi; fin2++) {
						if (fin1 == fin2) continue;
						cache = tex2D(xor2OutputPropLUT, REF2D(char,input,inpitch,tid,FIN(fans,goffset,fin1)), REF2D(char,input,inpitch,tid,FIN(fans,goffset,fin2)));
						pass += (cache > 1);
						tmp = tmp && (cache > 0);
						if (nfi > 1) {
							cache = tex2D(xor2InputPropLUT, REF2D(char,input,inpitch,tid,FIN(fans,goffset,fin1)), REF2D(char,input,inpitch,tid,FIN(fans,goffset,fin2)));
							fin = cache && fin && prev;
						}
					}
					REF2D(char,results,pitch,tid,FIN(fans,goffset,fin1)) = fin;
				}
//				resultCache = val && tmp && (pass <= nfi) && prev;
				break;
			default: break;
		}
		// stick the contents of resultCache into the results array

		*rowResults = resultCache;
	}
}

float gpuMarkPaths(GPU_Data& results, GPU_Data& input, GPU_Circuit& ckt) {
	HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting before we hit this point.
	loadPropLUTs();
	int startGate=ckt.size();
	int blockcount_y = (int)(results.block_width()/MARK_BLOCK) + (results.block_width()%MARK_BLOCK > 0);

#ifndef NTIMING
	float elapsed;
	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
#endif // NTIMING
	for (int i = ckt.levels(); i >= 0; i--) {
		dim3 numBlocks(ckt.levelsize(i),blockcount_y);
		startGate -= ckt.levelsize(i);
		kernMarkPathSegments<<<numBlocks,MARK_BLOCK>>>(input.gpu().data,input.gpu().pitch,results.gpu().data, ckt.gpu_graph(), ckt.offset(), results.width(), results.height(), startGate, results.gpu().pitch);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError()); // check to make sure we aren't segfaulting
	}
#ifndef NTIMING
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	elapsed = floattime(diff(start, stop));
	return elapsed;
#else 
	return 0.0;
#endif // NTIMING
}


void debugMarkOutput(ARRAY2D<char> results, std::string outfile) {
#ifndef NDEBUG
	char *lvalues;
	std::ofstream ofile(outfile.c_str());
	ofile << results.width << "," << results.height << std::endl;
	ofile << "Line:   \t";
	for (unsigned int i = 0; i < results.height; i++) {
		ofile << std::setw(OUTJUST) << i << " ";
	}
	ofile << std::endl;
	lvalues = (char*)malloc(results.height*results.pitch);
	cudaMemcpy2D(lvalues,results.pitch,results.data,results.pitch,results.width,results.height,cudaMemcpyDeviceToHost);
	for (unsigned int r = 0;r < results.width; r++) {
		ofile << "Vector " << r << ":\t";
		for (unsigned int i = 0; i < results.height; i++) {
			char z = REF2D(char, lvalues, results.pitch, r, i);
			switch(z) {
				case 0:
					ofile  << std::setw(OUTJUST+1) << "N "; break;
				case 1:
					ofile  << std::setw(OUTJUST+1) << "Y "; break;
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

