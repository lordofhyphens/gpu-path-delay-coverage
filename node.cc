#include "ckt.h"

NODEC::NODEC(std::string id) {
	this->initialize(id, 0, 0, 0, false, "");
}
NODEC::NODEC(std::string id, int type) {
	this->initialize(id, type, 0, 0, false, "");
}
NODEC::NODEC(std::string id, std::string type, int nfi, std::string finlist) {
	this->initialize(id, type, nfi, 0, false, finlist);
}
void NODEC::initialize(std::string id, int type, int nfi, int nfo, bool po, std::string finlist) {
	this->name = id;
	this->name.erase(std::remove_if(this->name.begin(), this->name.end(),isspace),this->name.end());
	this->placed = false;
	this->typ = type;
	this->level = 0;
	this->cur_fo = 1;
	this->nfi = nfi;
	this->nfo = nfo;
	this->po = po;
	this->finlist = finlist;
	this->finlist.erase(std::remove_if(this->finlist.begin(), this->finlist.end(),isspace),this->finlist.end());

	this->name.erase(std::remove_if(this->name.begin(), this->name.end(),isspace),this->name.end());
}
void NODEC::initialize(std::string id, std::string type, int nfi, int nfo, bool po, std::string finlist) {
	int n_type;
	type.erase(std::remove_if(type.begin(), type.end(),isspace),type.end());
	if (type == "NAND") {
		n_type = NAND;
	} else if (type == "AND") {
		n_type = AND;
	} else if (type == "OR") {
		n_type = OR;
	} else if (type == "NOR") {
		n_type = NOR;
	} else if (type == "DFF") {
		n_type = DFF;
	} else if (type == "XOR") {
		n_type = XOR;
	} else if (type == "XNOR") {
		n_type = XNOR;
	} else if (type == "FROM") {
		n_type = FROM;
	} else if (type == "NOT") {
		n_type = NOT;
	} else if (type == "BUFF") {
		n_type = BUFF;
	} else {
		n_type = UNKN;
	}
	this->initialize(id, n_type, nfi, nfo, po, finlist);
}

bool NODEC::operator==(const std::string& other) const {
	return this->name == other;
}
bool NODEC::operator==(const NODEC& other) const {
 // if a is not in the fanin of b and vice-versa, they are considered "equal"
	return (find_if(this->fin.begin(), this->fin.end(), StringFinder(other.name)) != this->fin.end()) && (find_if(other.fin.begin(),other.fin.end(), StringFinder(this->name)) != other.fin.end());
}
bool NODEC::operator<(const NODEC& other) const {
	// if a is in the fanin of b, then a < b
	return this->level < other.level;
}
std::ostream& operator<<(std::ostream& outstream, const NODEC& node) {
	outstream << node.name << "\t";
	switch (node.typ) {
		case INPT:
			outstream << "INPT\t"; break;
		case AND:
			outstream << "AND\t"; break;
		case NAND:
			outstream << "NAND\t"; break;
		case OR:
			outstream << "OR\t"; break;
		case NOR:
			outstream << "NOR\t"; break;
		case XNOR:
			outstream << "XNOR\t"; break;
		case BUFF:
			outstream << "BUFF\t"; break;
		case NOT:
			outstream << "NOT\t"; break;
		case FROM:
			outstream << "FROM\t"; break;
		case DFF:
			outstream << "DFF\t"; break;
		default:
			outstream << "UNKN\t";break;
	}
	outstream << (node.po == true ? "Yes" : "No" ) << "\t" << node.level << "\t" << node.nfi << "\t" << node.nfo << "\t" << node.finlist << "\t\t";
	std::vector< std::string >::iterator iter;
	for (unsigned int i = 0; i < node.fin.size(); i++) {
		outstream << node.fin[i].first;
		if (i != node.fin.size()-1)
			outstream << ",";
	}
	outstream << " | ";
	for (unsigned int i = 0; i < node.fot.size(); i++) {
		outstream << node.fot[i].first;
		if (i != node.fot.size()-1)
			outstream << ",";
	}
	outstream << std::endl;
	return outstream;
}

