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
	int mappings[ncount];

	for (int c = 0; c <= ncount; c++) {
		if (graph[c].typ == 0) {
			continue; // ignore blanks
		}
		pairiter pos = nodes.end(); // default to the front of the queue.
		LIST* tmp = NULL;
		for (pairiter i = nodes.begin(); i < nodes.end(); i++) {
			if (pos < i && pos != nodes.end()) continue;
			tmp = i->second.fin;
			while (tmp != NULL) {
				if (tmp->id == i->first)
					pos = i;
				tmp = tmp->nxt;
			}
		}

		pos = nodes.insert(pos, make_pair(c, graph[c]));
		assert((pos)->second.typ == graph[c].typ);
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
