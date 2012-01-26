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
	this->typ = type;
	this->level = 0;
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

bool NODEC::operator==(const NODEC& other) const {
	return this->name == other.name;
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
	outstream << (node.po == true ? "Yes" : "No" ) << "\t" << node.nfi << "\t" << node.nfo << "\t" << node.finlist << "\n";
	return outstream;
}

Circuit::Circuit() {
	this->graph = new std::vector<NODEC>();
}
Circuit::~Circuit() {
	delete this->graph;
}

void Circuit::read_bench(char* benchfile) {
	std::ifstream tfile(benchfile);
	this->name = benchfile;
	this->name.erase(std::remove_if(this->name.begin(), this->name.end(),isspace),this->name.end());
	this->name.erase(std::find(this->name.begin(),this->name.end(),'.'),this->name.end());
	std::vector<NODEC>* g = this->graph;
	std::string buffer, id;
	std::stringstream node;
	int front, back;
	while (getline(tfile,buffer)) {
		node.str(buffer);
		if (buffer.find("#") != std::string::npos) 
			continue;
		else if (buffer.find("INPUT") != std::string::npos) {
			front = buffer.find("(");
			back = buffer.find(")");
			id = buffer.substr(front+1, back - (front+1));
			g->push_back(NODEC(id, INPT));
		} else if (buffer.find("OUTPUT") != std::string::npos) {
			front = buffer.find("(");
			back = buffer.find(")");
			id = buffer.substr(front+1, back - (front+1));
			g->push_back(NODEC(id));
			g->back().po = true;
		} else if (buffer.find("=") != std::string::npos) {
			id = buffer.substr(0,buffer.find("="));
			front = buffer.find("(");
			back = buffer.find(")");
			std::string finlist = buffer.substr(front+1, back - (front+1));
			std::string gatetype = buffer.substr(buffer.find("=")+1,front - (buffer.find("=")+1));
			int nfi = count_if(finlist.begin(), finlist.end(), ispunct) + 1;
			if (find(g->begin(), g->end(), NODEC(id)) == g->end()) { 
				g->push_back(NODEC(id, gatetype, nfi, finlist));
			} else {
				// modify the pre-existing node. Node type should be unknown, and PO should be set.
				std::vector<NODEC>::iterator iter = find(g->begin(), g->end(), NODEC(id));
				assert(iter->po == true);
				assert(iter->typ == UNKN);
				*iter = NODEC(id, gatetype, nfi, finlist);
				iter->po = true;
			}
		} else {
			continue;
		}
	}
	for (std::vector<NODEC>::iterator iter = g->begin(); iter < g->end(); iter++) {
		node.str(iter->finlist);
		node.clear();
		while (getline(node,buffer,',')) {
			// figure out which which node has this as a fanout.
			std::vector<NODEC>::iterator j = find(g->begin(), g->end(), NODEC(buffer));
			j->nfo += 1;
			if (j->nfo > 1) {
			}
				iter->fin.push_back()
			std::cout << buffer << std::endl;
		}
	}


}
void Circuit::print() {
	std::vector<NODEC>* g = this->graph;
	std::cout << "Circuit: " << this->name << std::endl;
	std::cout << "Name\tType\tPO?\tNFI\tNFO" << std::endl;
	for (std::vector<NODEC>::iterator iter = g->begin(); iter < g->end(); iter++) {
		std::cout << *iter;
	}
}
