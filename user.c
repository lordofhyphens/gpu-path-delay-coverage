#include "project.h"
/****************************************************************************************************************************
Print all members of stuck structure
*****************************************************************************************************************************/
void PrintFault(FAULT *struck,int tot)
{
int  i;
printf("\nIndex\tNode\tStuckValue\n");
for(i=0;i<tot;i++){  printf("%d\t%d\t%d\n",i,struck[i].nod,struck[i].sval); }
return;
}//end of PrintFault
/****************************************************************************************************************************
Free the memory of all members of stuck structure
*****************************************************************************************************************************/
void ClearFault(FAULT *struck,int tot)
{
int  i;
for(i=0;i<tot;i++){  struck[i].nod=struck[i].sval=0; }
return;
}//end of ClearFault
/***************************************************************************************************************************
Read the fault file(.faults) and store it in struck structure
****************************************************************************************************************************/
int ReadFault(FILE *ffau,FAULT *struck)
{
int a,b,c,d;         //random variables
char *str,*itr;     //random variable
unsigned int m;     //random variable

str=(char *) malloc(Mlin * sizeof(char));           //dynamic memory allocation
itr=(char *) malloc(Mlin * sizeof(char));           //dynamic memory allocation
a=b=c=d=-1; m=1;
    
while(fgets(str,Mlin,ffau)!= NULL){    a=0;   
  while((str[a]!='\n')&&(str[a] !='\0')){ b++; 
     bzero(itr,Mlin); 
     for(c=0;str[a]!= '/';c++){  itr[c]=str[a];  a++;  }
     itr[c] = '\0'; struck[b].nod=atoi(itr);
     bzero(itr,Mlin); a++; 
     for(c=0;((str[a]!='\n')&&(str[a] !='\0'));c++){  itr[c]=str[a];  a++;  }
     itr[c] = '\0'; struck[b].sval=atoi(itr);  
  } } //end of file
free(itr); free(str);  //memory deallocation for dynamic arrays
return (b+1);
}//End of function
/***************************************************************************************************************************
Serial Fault Simulation
****************************************************************************************************************************/
void SerialFault(NODE *graph,int Max,PATTERN *vector,int Tot,FAULT *struck,int Tfs,FILE *fres,int Opt)
{
int a,b,c,d;  //random variables

for(c=0;c<3;c++){ 
   printf("\n\nX Value: %d",c);  fprintf(fres,"\n\nX Value: %d",c); 
  for(a=0;a<Tot;a++){
    printf("\n\nInput Pattern: %s",vector[a].piv);  fprintf(fres,"\n\nInput Pattern: %s",vector[a].piv); 
    SerialFaultFree(graph,Max,vector,a,fres,c);       // Fault free simulation of particular pattern
    for(b=0;b<Tfs;b++){
      SerialFaulty(graph,Max,struck[b].nod,struck[b].sval,fres,c); // Inject a fault and progate to output and check detection 
  } }}
printf("\n");  fprintf(fres,"\n"); 
return;
}//End of Serial Fault Simulation
/***************************************************************************************************************************
Apply a Single Input Vector on Primary Inputs and Propogate the values and Observe at Primary Outputs 
****************************************************************************************************************************/
void SerialFaultFree(NODE *graph,int Max,PATTERN *vector,int num,FILE *fres,int Opt)
{
int i,j,tmp,func,order=0;    //random variable
LIST *finInp=NULL;           //random variable

for(i=0;i<=Max;i++){ graph[i].val=2; } // Intialize the val=2 for all nodes
order=0;
for(i=0;i<=Max;i++){
  if(graph[i].typ!=0){  
    finInp=NULL; finInp=graph[i].fin;
    switch(graph[i].typ){
    case INPT:  
                if(vector[num].piv[order]=='1'){       graph[i].val=1; }
                else if(vector[num].piv[order]=='0'){  graph[i].val=0; }
                else if(vector[num].piv[order]=='x'){  graph[i].val=Opt; }
		order++; 
		break;
    case AND:	      
    case NAND:  func=1;
                for(j=0;j<graph[i].nfi&&finInp!=NULL;j++){   
                  if(Opt!=2){   tmp=1;
                    if((graph[finInp->id].val==0)||(func==0)){  tmp=0; }  }
                  else{  tmp=2;
                    if((graph[finInp->id].val==0)||(func==0)){       tmp=0; }
                    else if((graph[finInp->id].val==1)&&(func==1)){  tmp=1; }  }
                  func=tmp;  finInp=finInp->nxt; } // end for
                graph[i].val=func;
                if(graph[i].typ==NAND){ 
                  if(func!=2){   graph[i].val=1;  
                    if(func==1){ graph[i].val=0;  } } }
		break;
    case OR:	      
    case NOR:   func=0;
                for(j=0;j<graph[i].nfi&&finInp!=NULL;j++){   
                  if(Opt!=2){   tmp=0;
                    if((graph[finInp->id].val==1)||(func==1)){  tmp=1; } }
                   else{  tmp=2;
                     if((graph[finInp->id].val==1)||(func==1)){  tmp=1; } 
                     else if((graph[finInp->id].val==0)&&(func==0)){  tmp=0; }}
                  func=tmp;  finInp=finInp->nxt; } // end for
                graph[i].val=func;
                if(graph[i].typ==NOR){ 
                  if(func!=2){     graph[i].val=1;  
                    if(func==1){   graph[i].val=0; } } }
		break;
    case XOR:	      
    case XNOR:  func=0;
                for(j=0;j<graph[i].nfi&&finInp!=NULL;j++){   
                  if(Opt!=2){  tmp=1;
                    if(graph[finInp->id].val==func){    tmp=0; } }
                  else{  
                    if((graph[finInp->id].val==2)||(func==2)){ tmp=2; }
                    else{  tmp=1;
                    if(graph[finInp->id].val==func){    tmp=0; } } }
                  func=tmp;  finInp=finInp->nxt; } // end for
                graph[i].val=func;
                if(graph[i].typ==XNOR){  
                  if(func!=2){       graph[i].val=1; 
                    if(func==1){     graph[i].val=0; } } }
		break;
    case BUFF:	      
    case FROM:  graph[i].val=graph[finInp->id].val;            
		break;
    case NOT:   if(graph[finInp->id].val==1){       graph[i].val=0; }
                else if(graph[finInp->id].val==0){  graph[i].val=1; }
                else if(graph[finInp->id].val==2){  graph[i].val=2; }
		break;
    default:    printf("Invalid gate type %d for graph %d\n",graph[i].typ,i);
                break;   
     }// end switch	
   }//End of if(graph[i].type)	 
 }//End of for 
printf("\nInput Vector With X Assigned: "); fprintf(fres,"Input Vector With X Assigned: ");
for(i=1;i<=Max;i++){
  if((graph[i].typ!=0)&&(graph[i].nfi==0)){ printf("%d",graph[i].val); fprintf(fres,"%d",graph[i].val); } }
printf("\nFault Free: "); fprintf(fres,"\nFault Free: ");
for(i=1;i<=Max;i++){
  if(graph[i].po==1){ printf("%d",graph[i].val); fprintf(fres,"%d",graph[i].val); } }
return;
}//End of FaultFree
/***************************************************************************************************************************
Inject the Fault(Sfault) at particular node (Stnode) and Propagate the faulty values to Primary Output 
and Check the Fault is Detected or Not
****************************************************************************************************************************/
void SerialFaulty(NODE *graph,int Max,int Stnode,int Sfault,FILE *fres,int Opt)
{
int i,j,k,det,tmp,func;   //random variable
LIST *finInp=NULL;  //random variable

for(i=0;i<=Max;i++){ graph[i].mar=0; graph[i].fval=2; } // Intialize the mark=0 and fval=2 for all nodes
//Injecting the Sfault at Stnode and mark the node
graph[Stnode].fval=Sfault;   graph[Stnode].mar=1;  //printf("\nSfault %d at Stnode %d\n",Sfault,Stnode); 

//Propogate the fault values starting from next node to Stnode. 
//If the particular node is on the traversal path of Stnode mark it and propogate the fault value, otherwise use fault free value as faulty value
for(i=(Stnode+1);i<=Max;i++){
  if(graph[i].typ!=0){  
    finInp=NULL; finInp=graph[i].fin;
    switch(graph[i].typ){
    case INPT:  graph[i].fval=graph[i].val; 
		break;
    case AND:	      
    case NAND:  func=1; 
                for(j=0;j<graph[i].nfi&&finInp!=NULL;j++){      
                  if(Opt!=2){ tmp=1;
                    if(graph[finInp->id].mar==1){                    graph[i].mar=1;
                      if((graph[finInp->id].fval==0)||(func==0)){    tmp=0;             } }
                    else if((graph[finInp->id].mar==0)&&((graph[finInp->id].val==0)||(func==0))){ tmp=0;  }  }
                  else{ 
                    if(graph[finInp->id].mar==0){ tmp=2;
                      if((graph[finInp->id].val==0)||(func==0)){       tmp=0; }
                      else if((graph[finInp->id].val==1)&&(func==1)){  tmp=1; }  }
                    else if(graph[finInp->id].mar==1){ tmp=2; graph[i].mar=1;
                      if((graph[finInp->id].fval==0)||(func==0)){       tmp=0; }
                      else if((graph[finInp->id].fval==1)&&(func==1)){  tmp=1; }  } }
                  func=tmp;  finInp=finInp->nxt; } // end for
                graph[i].fval=func;
                if(graph[i].typ==NAND){ 
                  if(func!=2){        graph[i].fval=1;  
                    if(func==1){      graph[i].fval=0;  } } }
		break;
    case OR:	      
    case NOR: func=0; 
              for(j=0;j<graph[i].nfi&&finInp!=NULL;j++){      
                if(Opt!=2){ tmp=0;
                  if(graph[finInp->id].mar==1){                    graph[i].mar=1;
                    if((graph[finInp->id].fval==1)||(func==1)){    tmp=1;             } }
                  else if((graph[finInp->id].mar==0)&&((graph[finInp->id].val==1)||(func==1))){ tmp=1;  }  }
                else{ 
                  if(graph[finInp->id].mar==0){ tmp=2;
                    if((graph[finInp->id].val==1)||(func==1)){       tmp=1; }
                    else if((graph[finInp->id].val==0)&&(func==0)){  tmp=0; }  }
                  else if(graph[finInp->id].mar==1){ tmp=2; graph[i].mar=1;
                    if((graph[finInp->id].fval==1)||(func==1)){       tmp=1; }
                    else if((graph[finInp->id].fval==0)&&(func==0)){  tmp=0; }  } }
                  func=tmp;  finInp=finInp->nxt; } // end for
                graph[i].fval=func;
                if(graph[i].typ==NOR){ 
                  if(func!=2){        graph[i].fval=1;  
                    if(func==1){      graph[i].fval=0;  } } }
		break;  
    case XOR:	      
    case XNOR:  func=0;
                for(j=0;j<graph[i].nfi&&finInp!=NULL;j++){    
                  if(Opt!=2){  tmp=1;
                    if(graph[finInp->id].mar==1){          graph[i].mar=1; 
                      if(graph[finInp->id].fval==func){    tmp=0;           }}
                    else if((graph[finInp->id].mar==0)&&(graph[finInp->id].val==func)){tmp=0;  } }
                  else{  k=0; 
                    if(graph[finInp->id].mar==1){      graph[i].mar=1;  k=graph[finInp->id].fval;}
                    else if(graph[finInp->id].mar==0){ k=graph[finInp->id].val;  }
                    if((k!=2)&&(func!=2)){ tmp=1;
                      if(k==func){    tmp=0;  }}
                    else{ tmp=2;} }
                  func=tmp;  finInp=finInp->nxt; } // end for
                graph[i].fval=func;
                if(graph[i].typ==XNOR){ 
                  if(func!=2){        graph[i].fval=1;  
                    if(func==1){      graph[i].fval=0;  } } }
		break;
    case BUFF:	      
    case FROM:  if(graph[finInp->id].mar==1){     graph[i].fval=graph[finInp->id].fval;  graph[i].mar=1; }
                else if(graph[finInp->id].mar==0){graph[i].fval=graph[finInp->id].val;                   }         
		break;
    case NOT:   if(graph[finInp->id].mar==1){      k=graph[finInp->id].fval; graph[i].mar=1; }
                else if(graph[finInp->id].mar==0){ k=graph[finInp->id].val;                  } 
                graph[i].fval=2;
                if(k==0){       graph[i].fval=1; }
                else if(k==1){  graph[i].fval=0; } 
		break;
    default:    printf("Invalid gate type %d for graph %d\n",graph[i].typ,i);
                break;   
     } // end switch	
   }//End of if(graph[i].type)	 
 }//End of for 
det=0;
for(i=1;i<=Max;i++){
  if((graph[i].po==1)&&(graph[i].mar==1)&&(graph[i].val!=graph[i].fval)){  det=1; } } 

if(det==1){
  printf("\nDetected : %d/%d",Stnode,Sfault);  fprintf(fres,"\nDetected : %d/%d",Stnode,Sfault);
  printf("\nFaulty Response: "); fprintf(fres,"\nFaulty Response: ");
  for(i=1;i<=Max;i++){
    if(graph[i].po==1){ printf("%d",graph[i].fval); fprintf(fres,"%d",graph[i].fval); } }  }
return;
}//End of CreateFaulty
/****************************************************************************************************************************/
