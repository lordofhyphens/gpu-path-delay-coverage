#include "ckt.h"

typedef std::vector<NODEC>::iterator nodeiter;
Circuit::Circuit() {
	this->graph = new std::vector<NODEC>();
	this->_levels = 1;
}
Circuit::~Circuit() {
	delete this->graph;
}
void Circuit::save(char* memfile) {
	std::ofstream ofile(memfile);
	
}
void Circuit::load(char* memfile) {
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
				nodeiter iter = find(g->begin(), g->end(), id);
				assert(iter->po == true);
				assert(iter->typ == UNKN);
				*iter = NODEC(id, gatetype, nfi, finlist);
				iter->po = true;
			}
		} else {
			continue;
		}
	}
	for (nodeiter iter = g->begin(); iter < g->end(); iter++) {
		if (iter->finlist == "")
			continue;
		node.str(iter->finlist);
		node.clear();
		while (getline(node,buffer,',')) {
			// figure out which which node has this as a fanout.
			nodeiter j = find(g->begin(), g->end(), buffer);
			j->nfo++;
		}
	}
	std::vector<NODEC> temp_batch;
	for (nodeiter iter = g->begin(); iter < g->end(); iter++) {
		node.str(iter->finlist);
		node.clear();
		std::string newfin = "";
		while (getline(node,buffer,',')) {
			nodeiter j = find(g->begin(), g->end(), buffer);
			if (j->nfo < 2) {
				iter->fin.push_back(std::make_pair(j->name, -1));
				j->fot.push_back(std::make_pair(iter->name, -1));
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
				temp_batch.back().fot.push_back(std::make_pair(iter->name,-1));
				temp_batch.back().fin.push_back(std::make_pair(j->name,-1));
				temp_batch.back().nfo = 1;
				j->fot.push_back(std::make_pair(temp_batch.back().name,-1));
				iter->fin.push_back(std::make_pair(j->name+"fan"+tmp.str(),-1));
				if (newfin == "") {
					newfin += j->name+"fan"+tmp.str();
				} else {
					newfin += "," + j->name+"fan"+tmp.str();
				}
			}
		}
		iter->finlist = newfin;
	}
	for (nodeiter iter = temp_batch.begin(); iter < temp_batch.end(); iter++) {
		g->push_back(*iter);
	}
	remove_if(g->begin(),g->end(),isUnknown);
	this->levelize();
	std::sort(g->begin(), g->end());
	annotate();
}
bool isPlaced(const NODEC& node) {
	return (node.placed == 0);
}
inline bool Yes(const NODEC& node) {
	return true;
}
// levelize the circuit.
void Circuit::levelize() {
	std::vector<NODEC>* g = this->graph;

	while (count_if(g->begin(),g->end(), isPlaced) > 0) {
		for (nodeiter iter = g->begin(); iter < g->end(); iter++) {
			if (iter->placed == false) {
				if (iter->typ == INPT)  {
					iter->level = 0;
					iter->placed = true;
				} else {
					bool allplaced = true;
					int level = 0;
					for (unsigned int i = 0; i < iter->fin.size(); i++) {
						allplaced = allplaced && find(g->begin(), g->end(), iter->fin[i].first)->placed;
						if (level < find(g->begin(), g->end(), iter->fin[i].first)->level) 
							level = find(g->begin(), g->end(), iter->fin[i].first)->level;
					}
					if (allplaced == true) { 
						iter->level = ++level;
						iter->placed = true;
						if (level+1 > this->_levels)
							this->_levels = level;
					}
				}
			}
		}
	}
}
void Circuit::print() const {
	std::vector<NODEC>* g = this->graph;
	std::cout << "Circuit: " << this->name << std::endl;
	std::cout << "Name\tType\tPO?\tLevel\tNFI\tNFO\tFinlist\t\tFin | Fot" << std::endl;
	for (nodeiter iter = g->begin(); iter < g->end(); iter++) {
		std::cout << *iter;
	}
}

int Circuit::levelsize(int l) const {
	return countInLevel(*graph, l);
}
// labels each fanin of each circuit 
void Circuit::annotate() {
	std::vector<NODEC>* g = this->graph;
	for (std::vector<NODEC>::iterator iter = g->begin(); iter < g->end(); iter++) {
		for (std::vector<std::pair<std::string, int> >::iterator i = iter->fin.begin(); i < iter->fin.end(); i++) {
			i->second = count_if(g->begin(), find(g->begin(),g->end(),i->first), Yes);
		}
		for (std::vector<std::pair<std::string, int> >::iterator i = iter->fot.begin(); i < iter->fot.end(); i++) {
			i->second = count_if(g->begin(), find(g->begin(),g->end(),i->first), Yes);
		}
	}
}
int countInLevel(std::vector<NODEC>& v, int level)  {
	int cnt = 0;
	for (std::vector<NODEC>::iterator iter = v.begin(); iter < v.end(); iter++) {
		if (isInLevel(*iter, level)) 
			cnt++;
	}
	return cnt;
}
bool isUnknown(const NODEC& node) {
	return node.typ == UNKN;
}
bool isInLevel(const NODEC& node, int N) {
	return node.level == N;
}
