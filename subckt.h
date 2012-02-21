#ifndef SUBCKT_H
#define SUBCKT_H
#include <vector>
#include <algorithm>
#include <string>
#include "ckt.h"
class SubCkt { 
	private:
		const Circuit& _ckt;
		int _ref_node;
		std::vector<int>* _levels;
		std::vector<int>* _subckt;

		void grow_recurse_back(unsigned int node);
		void grow_recurse_forward(unsigned int node);
	public:	
		~SubCkt();
		std::string save();
		void load(const std::string& memfile);
		SubCkt(const Circuit& ckt);
		SubCkt(const SubCkt&);
		SubCkt(const Circuit& ckt, unsigned int node);
		void add(const int& n) { add(this->_ckt, n);}
		void add(const Circuit&, const int&);
		int* gpu(); // get the flat array representation, allocates memory.
		int at(unsigned int);
		int levelsize(unsigned int);
		std::vector<int>& subckt() const { return *_subckt; }
		const SubCkt operator/(const SubCkt& b) const; // intersection
		SubCkt& operator=(const SubCkt&);
		void grow(unsigned int node);
};

#endif // SUBCKT_H
