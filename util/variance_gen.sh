#!/bin/bash

for i in `seq 1 1 30`; do ./fcount $1 $2 2> log-$i.txt; echo "$i of 30"; done
for i in `seq 1 1 30`; do for j in `seq 1 1 30`; do diff -q log-$i.txt log-$j.txt; done;done
