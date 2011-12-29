#include <deque>
#include <vector>
#include <utility>
#include "defines.h"
#include "iscas.h"
using namespace std;

// Memory-expensive topological sort using a deques (used mostly as a list here
// because of the [] deque has.)

typedef pair<int, NODE> nodepair; 
typedef deque<nodepair>::iterator pairiter;

int topologicalSort(NODE *graph, int ncount) {
	deque<nodepair> nodes;
	int ncnt = 0;
	int mappings[ncount];
	for (int i = 0; i <= ncount;i++){
		if (graph[i].typ != 0)
			ncnt++;
	}
	while (nodes.size() < (unsigned int)ncnt) {
		for (int c = 0; c <= ncount; c++) {
			if (graph[c].typ == 0) {
				continue; // ignore blanks
			}
			if (graph[c].typ == INPT) {
				nodes.push_back(make_pair(c, graph[c]));
				graph[c].typ = 0;
				continue;
			}
			LIST* fin = graph[c].fin;
			int supported = 0;
			while (fin != NULL) {
				for (pairiter i = nodes.begin(); i< nodes.end(); i++) {
					if(fin->id == i->first) {
						supported += 1;
						break;
					}
				}
				fin = fin->nxt;
			}
			if (supported >= graph[c].nfi) {
				nodes.push_back(make_pair(c, graph[c]));
				graph[c].typ = 0;
				continue;
			}
		}
	}
	int j = 0;
	for (pairiter i = nodes.begin(); i < nodes.end(); i++) {
		mappings[i->first] = j;
		j++; 
	}
	LIST* tmp = NULL;
	for (pairiter i = nodes.begin(); i < nodes.end(); i++) {
		tmp = i->second.fin;
		while (tmp != NULL) {
			tmp->id = mappings[tmp->id];
			tmp = tmp->nxt;
		}
		tmp = i->second.fot;
		while (tmp != NULL) {
			tmp->id = mappings[tmp->id];
			tmp = tmp->nxt;
		}
	}
	j = 0;
	for (pairiter i = nodes.begin(); i < nodes.end(); i++) {
		graph[j] = i->second;
		assert(graph[j].typ == i->second.typ);
		assert(graph[j].typ != 0);
		j++;

 	}
	return nodes.size();
}
