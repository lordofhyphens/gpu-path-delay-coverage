#include "vectors.h"

/* Read a simple text file formatted with input patterns.
* This modifies the array in vecs, allocating it. 
* It returns the count of input patterns. All don'tcares 
* are set to '0'.
*/
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
		for (unsigned int j = 0; j < str1.size(); j++) { 
			REF2D(char, pack.cpu(chunk),pack.pitch(),lines, j) = str1[j];
		}
		lines++;
		if (lines > chunksize) {
			lines = 0;
			chunk++;
		}
	}
	return ERR_NONE;
}

