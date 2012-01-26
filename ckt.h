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

#define S0 0
#define S1 1
#define T0 2 // 1->0 transition, or "Transition to 0"
#define T1 3 // 0->1 transition, or "Transition to 1"

struct NODEC {
	std::string name;
	char typ;
	int nfi, nfo, level;
	bool po;
	std::string finlist;
	std::list<int> fin;
	std::list<int> fot;
	NODEC(std::string);
	NODEC(std::string, int type);
	NODEC(std::string id, std::string type, int nfi, std::string finlist);
	bool operator==(const NODEC& other) const;
	private:
		void initialize(std::string id, int type, int nfi, int nfo, bool po, std::string finlist);
		void initialize(std::string id, std::string type, int nfi, int nfo, bool po, std::string finlist);
};

class Circuit {
	private:
		std::vector<NODEC>* graph;
		std::string name;
		void sort();
		void mark_lines();
	public:
		Circuit();
		~Circuit();
		void read_bench(char* benchfile);
		void print();
};

std::ostream& operator<<(std::ostream& outstream, const NODEC& node);
#endif //CKT_H
