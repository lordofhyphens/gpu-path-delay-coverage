#include "ckt.h"

Circuit::Circuit() {
	this->graph = new std::vector<NODEC>();
	this->_levels = 1;
}
Circuit::~Circuit() {
	delete this->graph;
}
int Circuit::levels() { 
	return this->_levels;
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
			id.erase(std::remove_if(id.begin(), id.end(),isspace),id.end());
			front = buffer.find("(");
			back = buffer.find(")");
			std::string finlist = buffer.substr(front+1, back - (front+1));
			std::string gatetype = buffer.substr(buffer.find("=")+1,front - (buffer.find("=")+1));
			int nfi = count_if(finlist.begin(), finlist.end(), ispunct) + 1;
			if (find(g->begin(), g->end(), id) == g->end()) { 
				g->push_back(NODEC(id, gatetype, nfi, finlist));
			} else {
				// modify the pre-existing node. Node type should be unknown, and PO should be set.
				std::vector<NODEC>::iterator iter = find(g->begin(), g->end(), id);
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
		if (iter->finlist == "")
			continue;
		node.str(iter->finlist);
		node.clear();
		while (getline(node,buffer,',')) {
			// figure out which which node has this as a fanout.
			std::vector<NODEC>::iterator j = find(g->begin(), g->end(), buffer);
			j->nfo++;
		}
	}
	std::vector<NODEC> temp_batch;
	for (std::vector<NODEC>::iterator iter = g->begin(); iter < g->end(); iter++) {
		node.str(iter->finlist);
		node.clear();
		std::string newfin = "";
		while (getline(node,buffer,',')) {
			std::vector<NODEC>::iterator j = find(g->begin(), g->end(), buffer);
			if (j->nfo < 2) {
				iter->fin.push_back(j->name);
				j->fot.push_back(iter->name);
				if (newfin == "") {
					newfin += j->name;
				} else {
					newfin += "," + j->name;
				}
			} else {
				std::stringstream tmp;
				tmp << j->cur_fo;
				j->cur_fo+=1;
				temp_batch.push_back(NODEC((j->name+"fan"+tmp.str()),"FROM",1,j->name));
				temp_batch.back().fot.push_back(iter->name);
				temp_batch.back().fin.push_back(j->name);
				temp_batch.back().nfo = 1;
				j->fot.push_back(temp_batch.back().name);
				iter->fin.push_back(j->name+"fan"+tmp.str());
				if (newfin == "") {
					newfin += j->name+"fan"+tmp.str();
				} else {
					newfin += "," + j->name+"fan"+tmp.str();
				}
			}
		}
		iter->finlist = newfin;
	}
	for (std::vector<NODEC>::iterator iter = temp_batch.begin(); iter < temp_batch.end(); iter++) {
		g->push_back(*iter);
	}
	this->levelize();
	std::stable_sort(g->begin(), g->end());
}
bool isPlaced(const NODEC& node) {
	return (node.placed == 0);
}
// levelize the circuit.
void Circuit::levelize() {
	std::vector<NODEC>* g = this->graph;
	std::vector<NODEC> dest;
	while (count_if(g->begin(),g->end(), isPlaced) > 0) {
		for (std::vector<NODEC>::iterator iter = g->begin(); iter < g->end(); iter++) {
			if (iter->placed == false) {
				if (iter->typ == INPT)  {
					iter->level = 0;
					iter->placed = true;
					continue;
				}
				bool allplaced = true;
				int level = 0;
				for (unsigned int i = 0; i < iter->fin.size(); i++) {
					if (find(g->begin(), g->end(), iter->fin[i])->placed != true) {
						allplaced = false;
					} else {
						if (level < find(g->begin(), g->end(), iter->fin[i])->level) 
							level = find(g->begin(), g->end(), iter->fin[i])->level;
					}
				}
				if (allplaced == true) { 
					iter->level = level+1;
					iter->placed = true;
					if (level+1 > this->_levels)
						this->_levels = level+1;
				}
			}
		}
	}
}
void Circuit::print() {
	std::vector<NODEC>* g = this->graph;
	std::cout << "Circuit: " << this->name << std::endl;
	std::cout << "Name\tType\tPO?\tLevel\tNFI\tNFO\tFinlist\t\tFin | Fot" << std::endl;
	for (std::vector<NODEC>::iterator iter = g->begin(); iter < g->end(); iter++) {
		std::cout << *iter;
	}
}
