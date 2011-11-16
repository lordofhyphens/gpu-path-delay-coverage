#include "iscas.h"
int counter = 1;
/*****************************************************
 * Insert an element "x" at end of LIST "l", if "x" is not already in "l".
 *****************************************************/
void InsertList(LIST **l,int x,int lineid)
{
    LIST *p,*tl;
    if ((p=(LIST *) malloc(sizeof(LIST)))==NULL) {
        printf("LIST: Out of memory\n");
        exit(1);
    }
    else {
        p->id=x;
		p->line=lineid;
        p->nxt=NULL;
        if(*l==NULL) {
            *l=p;
            return;
        }
        tl=NULL;
        tl=*l;
        while(tl!=NULL) {
            if(tl->id==x) {
                break;
            }
            if(tl->nxt==NULL) {
                tl->nxt=p;
            }
            tl = tl->nxt;
        }
    }
    return;
}//end of InsertList
/*****************************************************
 * Print the elements in LIST "l"
 *****************************************************/
void PrintList(LIST *l)
{
    LIST *temp=NULL;

    temp=l;
    while(temp!=NULL) {
        printf("(%d,%d) ", temp->id, temp->line);
        temp = temp->nxt;
    }
    return;
}//end of PrintList
/*****************************************************
 * Free all elements in  LIST "l"
 *****************************************************/
void FreeList(LIST **l)
{
    LIST *temp=NULL;

    if(*l==NULL) {
        return;
    }
    temp=(*l);
    while((*l) != NULL) {
        temp=temp->nxt;
        free(*l);
        (*l)=temp;
    }
    (*l)=NULL;
    return;
}//end of FreeList
/******************************************************
 * Routine to read the Bench Mark(.isc files)
 ******************************************************/
int ReadIsc(FILE *fisc,NODE *graph)
{
    char c,noty[Mlin],from[Mlin],str1[Mlin],str2[Mlin],name[Mlin],line[Mlin];
    int  i,id,fid,fin,fout,mid=0;

    // intialize all nodes in graph structure
    for(i=0; i<Mnod; i++) {
        InitializeCircuit(graph,i);
    }
    //skip the comment lines
    do
        fgets(line,Mlin,fisc);
    while(line[0] == '*');
    // read line by line
    while(!feof(fisc)) {
        //initialize temporary strings
        bzero(noty,strlen(noty));
        bzero(from,strlen(from));
        bzero(str1,strlen(str1));
        bzero(str2,strlen(str2));
        bzero(name,strlen(name));
        //break line into data
        sscanf(line, "%d %s %s %s %s",&id,name,noty,str1,str2);
        //fill in the type
        strcpy(graph[id].nam,name);
        graph[id].typ=AssignType(noty);
        //fill in fanin and fanout numbers
        if(graph[id].typ!=FROM) {
            fout= atoi(str1);
            fin=atoi(str2);
        }
        else {
            fin=fout= 1;
            strcpy(from,str1);
        }
        if(id > mid) {
            mid=id;
        }
        graph[id].nfo=fout;
        graph[id].nfi=fin;
        if(fout==0) {
            graph[id].po=1;
        }
        //create fanin and fanout lists
        switch (graph[id].typ)  {
        case 0     :
            printf("ReadIsc: Error in input file (Node %d)\n",id);
            exit(1);
        case INPT  :
            break;
        case AND   :
        case NAND  :
        case OR    :
        case NOR   :
        case XOR   :
        case XNOR  :
        case BUFF  :
        case NOT   :
            for(i=1; i<=fin; i++) {
                fscanf(fisc, "%d", &fid);
                InsertList(&graph[id].fin,fid,-1);
                InsertList(&graph[fid].fot,id, -1);
            }
            fscanf(fisc,"\n");
            break;
        case FROM  :
            for(i=mid; i>0; i--) {
                if(graph[i].typ!=0) {
                    if(strcmp(graph[i].nam,from)==0) {
                        fid=i;
                        break;
                    }
                }
            }
            InsertList(&graph[id].fin,fid,-1);
            InsertList(&graph[fid].fot,id,-1);
            break;
        } //end case
        bzero(line,strlen(line));
        fgets(line,Mlin,fisc);
    } // end while
    return mid;
}//end of ReadIsc
/******************************************************
 * Initialize the paricular member of graph structure
 ******************************************************/
void InitializeCircuit(NODE *graph,int num)
{
    bzero(graph[num].nam,Mnam);
    graph[num].typ=graph[num].nfi=graph[num].nfo=graph[num].po=graph[num].mar=0;
    graph[num].val=graph[num].fval=2;
    graph[num].fin=graph[num].fot=NULL;
    return;
}
void InitializeLines(LINE *graph,int num)
{
    graph[num].logic=graph[num].prev=graph[num].next=-1;
    return;
}
/******************************************************
 * Convert (char *) type read to (int)
 ******************************************************/
int AssignType(char *typ) {
    if      ((strcmp(typ,"inpt")==0) ||
             (strcmp(typ,"INPT")==0))       return 1;
    else if ((strcmp(typ,"and")==0)  ||
             (strcmp(typ,"AND")==0))        return 2;
    else if ((strcmp(typ,"nand")==0) ||
             (strcmp(typ,"NAND")==0))       return 3;
    else if ((strcmp(typ,"or")==0)   ||
             (strcmp(typ,"OR")==0))         return 4;
    else if ((strcmp(typ,"nor")==0)  ||
             (strcmp(typ,"NOR")==0))        return 5;
    else if ((strcmp(typ,"xor")==0)  ||
             (strcmp(typ,"XOR")==0))        return 6;
    else if ((strcmp(typ,"xnor")==0) ||
             (strcmp(typ,"XNOR")==0))       return 7;
    else if ((strcmp(typ,"buff")==0) ||
             (strcmp(typ,"BUFF")==0))       return 8;
    else if ((strcmp(typ,"not")==0)  ||
             (strcmp(typ,"NOT")==0))        return 9;
    else if ((strcmp(typ,"from")==0) ||
             (strcmp(typ,"FROM")==0))       return 10;
    else return 0;
}

/******************************************************
 * Print all members of graph structure(except typ=0)
 * after reading the bench file
 ******************************************************/
void PrintCircuit(NODE *graph,int Max)
{
    LIST *temp;
    int  i;
    printf("\nID\tNAME\tTYPE\tPO\tIN#\tOUT#\tVAL\tFVAL\tMARK\tFANIN\tFANOUT\n");
    for(i=0; i<=Max; i++) {
        if(graph[i].typ!=0) {
            printf("%d\t%s\t%d\t%d\t%d\t%d\t",i,graph[i].nam,graph[i].typ,graph[i].po,graph[i].nfi,graph[i].nfo);
            printf("%d\t%d\t%d\t",graph[i].val,graph[i].fval,graph[i].mar);
            temp=NULL;
            temp=graph[i].fin;
            if(temp!=NULL) {
                PrintList(temp);
            }
            printf("\t");
            temp=NULL;
            temp=graph[i].fot;
            if(temp!=NULL) {
                PrintList(temp);
            }
            printf("\n");
        }
    }
    return;
}

/******************************************************
 * Free the memory of all member of graph structure
 ******************************************************/
void ClearCircuit(NODE *graph,int i)
{
    int num=0;
    for(num=0; num<i; num++) {
        graph[num].typ=graph[num].nfi=graph[num].nfo=graph[num].po=0;
        graph[num].mar=graph[num].val=graph[num].fval=0;
        if(graph[num].typ!=0) {
            bzero(graph[num].nam,Mnam);
            if(graph[num].fin!=NULL) {
                FreeList(&graph[num].fin);
                graph[num].fin = NULL;
            }
            if(graph[num].fot!=NULL) {
                FreeList(&graph[num].fot);
                graph[num].fot = NULL;
            }
        }
    }
    return;
}
// Enumerate (label) every line in the circuit. 
// Returns the number of lines in the circuit.
int EnumerateLines(NODE *graph, LINE *lgraph) {
	// lid is -1 because the algorithm increments before assigning.
	int lid = -1, init;  
	for (int i = 0; i < Mnod; i++) {
		if (graph[i].typ == 0) 
			continue;
		LIST* cur = graph[i].fin;
		LIST* tmp = NULL;
		while (cur != NULL) {
			if (cur->line == -1) {
				tmp = graph[cur->id].fot;
				while (tmp != NULL) {
					if (tmp->id == i) {
						cur->line = tmp->line;
						break;
					}
					tmp = tmp->nxt;
				}

			}
			cur = cur->nxt;
		}
		cur = graph[i].fot;
		tmp = NULL;
		init = 1;
		if (cur == NULL) {
			// gate is primary output
			lid++;
//			printf("%d\n",lid);
			lgraph[lid].prev = i;
		} else {
			while (cur != NULL) {
				if (graph[cur->id].typ == FROM) {
					if (init == 1) {
						lid++;
						init = 0;
					}
				} else { 
					lid++;
				}
				cur->line = lid;
				lgraph[lid].prev = i;
				lgraph[lid].next = cur->id;
//				printf ("%d,%d\t",lid, cur->id);
				cur = cur->nxt;
			}
		}
//		printf("\n");
	}
	return lid+1;
}
void PrintLines(LINE* lgraph, int lcnt) {
	printf("ID\tVALUE\tPREV\tNEXT\n");
	for (int i = 0; i < lcnt; i++) {
		printf("%d:\t%d\t%d\t%d\n",i,lgraph[i].logic,lgraph[i].prev,lgraph[i].next);
	}
}
