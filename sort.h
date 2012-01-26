#ifndef SORT_H
#define SORT_H
#include <deque>
#include "iscas.h"
int topologicalSort(NODE *graph, int ncount);
int levelSort(NODE* graph, int ncount);
int levelize(NODE* graph, int ncount);
#endif
