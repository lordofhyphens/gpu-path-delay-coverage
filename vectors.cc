#include "vectors.h"

/* Read a simple text file formatted with input patterns.
* This modifies the array in vecs, allocating it. 
* It returns the count of input patterns. All don'tcares 
* are set to '0'.
*/
std::pair<size_t, size_t> get_vector_dim(char* fvec) {
	std::string str1;
	std::ifstream tfile(fvec);
	size_t lines = 0;
	size_t inputs = 0;
	while(getline(tfile,str1)) {
		lines++;
		inputs = str1.size();
	}
	tfile.close();
	return std::make_pair<size_t,size_t>(lines, inputs);
}
int read_vectors(GPU_Data& pack, char* fvec, int chunksize) {
	std::string str1;
	std::ifstream tfile(fvec);
	int chunk = 0;
	int lines = 0;
	while(getline(tfile,str1)) {
		if (str1.find("#") != std::string::npos) 
			continue; // ignore comment lines
		// for every character in the string, 
		// determine the placement in the array, using
		// REF2D.
//		std::cout << str1 << std::endl;
		for (unsigned int j = 0; j < str1.size(); j++) { 
			REF2D(char, pack.cpu(chunk).data,pack.cpu(chunk).pitch,lines, j) = ((str1[j] == '0') ? 0 : 1);
//			DPRINT("%2d ",REF2D(char, pack.cpu(chunk).data,pack.cpu().pitch,lines, j) );
		}
		lines++;
		if (lines > chunksize) {
			lines = 0;
			chunk++;
		}
	}
	std::cerr << " All vectors have been read." << std::endl;
	tfile.close();
	pack.refresh();
	return ERR_NONE;
}
