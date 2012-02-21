#ifndef SUBCKT_H
#define SUBCKT_H
#include <vector>
class SubCkt { 
	private:
		int* g_subckt; // stored on the GPU.
		std::vector<int>* levels;
		std::vector<int>* subckt;
	public:	
		void add(const int&);
		void add(const NODEC&);
		int* gpu(); // get the flat array representation.
}

#endif // SUBCKT_H
