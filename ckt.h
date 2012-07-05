#ifndef CKT_H
#define CKT_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <cassert>
#include <utility>
#include <stdint.h>
#include "defines.h"

#define FBENCH 1 // iscas89 BENCH format.
// NODE TYPE CONSTANTS 
#define UNKN 0				// unknown node
#define INPT 1				// Primary Input
#define AND  2				// AND 
#define NAND 3				// NAND 
#define OR   4				// OR 
#define NOR  5				// NOR 
#define XOR  6				// XOR 
#define XNOR 7				// XNOR 
#define BUFF 8				// BUFFER 
#define NOT  9				// INVERTER 
#define FROM 10				// STEM BRANCH
#define DFF 11				// Dflipflop

struct NODEC {
	std::string name;
	uint8_t typ;
	uint32_t nfi, nfo, level;
	uint32_t cur_fo;
	bool po, placed;
	std::string finlist;
	std::vector<std::pair<std::string, uint32_t > > fin;
	std::vector<std::pair<std::string, uint32_t > > fot;
	NODEC() { name = "", typ = 0, nfi = 0, nfo = 0, po = false, finlist="";}
	NODEC(std::string);
	NODEC(std::string, uint8_t type);
	NODEC(std::string id, std::string type, uint32_t nfi, std::string finlist);
	bool operator==(const std::string& other) const;
	bool operator==(const NODEC& other) const;
	bool operator<(const NODEC& other) const;
	bool operator>(const NODEC& other) const;
	bool operator<=(const NODEC& other) const;
	bool operator>=(const NODEC& other) const;
	private:
		void initialize(std::string id, uint8_t type, uint32_t nfi, uint32_t nfo, bool po, std::string finlist);
		void initialize(std::string id, std::string type, uint32_t nfi, uint32_t nfo, bool po, std::string finlist);
		void load(std::string attr);
};

class Circuit {
	protected:
		std::vector<NODEC>* graph;
		std::string name;
		void levelize();
		void mark_lines();
		uint32_t _levels;
		void annotate();
	public:
		Circuit();
		Circuit(uint8_t type, const char* benchfile) {
			this->graph = new std::vector<NODEC>();
			this->_levels = 1;
			if (type == FBENCH)
				this->read_bench(benchfile);
		}
		~Circuit();
		bool nodelevel(uint32_t n, uint32_t m) const;
		void read_bench(const char* benchfile);
		void print() const;
		NODEC& at(uint32_t node) const { return this->graph->at(node);}
		inline uint32_t levels() const { return this->_levels;}
		uint32_t levelsize(uint32_t) const;
		uint32_t size() const { return this->graph->size();}
		void save(const char*); // save a copy of the circuit in its current levelized form
		void load(const char* memfile); // load a circuit that has been levelized.
};

std::ostream& operator<<(std::ostream& outstream, const NODEC& node);
bool isPlaced(const NODEC& node);
bool isInLevel(const NODEC& node, uint32_t N);

uint32_t countInLevel(std::vector<NODEC>& v, uint32_t level);
bool isUnknown(const NODEC& node) ;
bool isDuplicate(const NODEC& a, const NODEC& b);
bool nameSort(const NODEC& a, const NODEC& b);

struct StringFinder
{
  StringFinder(const std::string & st) : s(st) { }
  const std::string s;
  bool operator()(const std::pair<std::string, int>& lhs) const { return lhs.first == s; }
};


template <class T>
bool from_string(T& t, const std::string& s, std::ios_base& (*f)(std::ios_base&))
{
  std::istringstream iss(s);
  return !(iss >> f >> t).fail();
}
#endif //CKT_H
