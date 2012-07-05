#ifndef SUBCKT_H
#define SUBCKT_H
#include <vector>
#include <algorithm>
#include <string>
#include <iterator>
#include "ckt.h"
class SubCkt { 
	private:
		const Circuit& _ckt;
		int _ref_node;
		int *_flat;
		int *_gpu;
		int* flat(); // get the flat array representation, allocates memory.
		std::vector<int>* _levels;
		std::vector<int>* _subckt;
		void levelize();

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
		void copy();
		void clear();
		bool operator<(const SubCkt&) const;
		bool operator<(const int) const;

		int* gpu() { return this->_gpu;}
		int at(unsigned int);
		int in(unsigned int);
		int levelsize(unsigned int);
		int levels();
		int size() const { return this->_subckt->size();}
		std::vector<int>& subckt() const { return *_subckt; }
		const SubCkt operator/(const SubCkt& b) const; // intersection
		SubCkt& operator=(const SubCkt&);
		void grow(unsigned int node);
};

#endif // SUBCKT_H
