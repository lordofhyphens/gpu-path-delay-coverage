#!/usr/bin/python
import argparse
import re
parser = argparse.ArgumentParser(description='Convert ISCAS89 to ISCAS85')
parser.add_argument('benchmark', metavar='IN', type=str,
                   help='input')
parser.add_argument('output', metavar='OUT', type=str, help='output')
args = parser.parse_args()

def f5(seq, idfun=None): 
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

def f2(seq): 
    # order preserving
    checked = []
    for e in seq:
        if e not in checked:
            checked.append(e)
    return checked
f = open(args.benchmark, 'r')
outfile = open(args.output, 'w')
ckt = list()
seen = list()
sortckt = list()
for j in f:
	t = re.sub("\n","",j)
	r = re.split("#",t,2)
	t = r[0]
	if len(r) > 1:
		comment = r[1]
	else:
		comment = "";
	if len(r[0]) > 1:
		if re.search("INPUT", t): 
			tmp = re.sub("INPUT\(","",t)
			tmp = re.sub("\)","",tmp)
			seen.append(tmp)
			ckt.append([tmp, [], j])
			continue;
		elif re.search("OUTPUT", t):
			ckt.append(["", [], j])
		elif re.search("=",t):
			ar = re.split("=", j, 2)
			out = re.sub(" ", "", ar[0]);
			ar = ar[1]
			ar = re.split("\(",ar,2)
			ind = -1;
			ar[1] = re.sub("\n","",ar[1])
			dep = re.split(",", re.sub("[() ]","",ar[1]))
			ckt.append([out, dep, j])
			continue;
	else:
		ckt.append(["", [], j])
p = 0
currcount = 0
gatecount = len(ckt) - len([x for x in ckt if x[1] == []])
while (len(sortckt) < len(ckt)):
	for i in ckt:
		if i[1] == []:
			if p < 1:
				sortckt.append(i)
		elif i not in sortckt:
			sortpos = -1
			count = 0
			for j in i[1]:
				if j in [x[0] for x in sortckt]:
					# need to place after.
					if sortpos < [x[0] for x in sortckt].index(j):
						sortpos = [x[0] for x in sortckt].index(j)+1
					count += 1
			if sortpos > 0 and count == len(i[1]):
				print "Placing gate", j, "\t\t", currcount, "of", gatecount, "gates."
				sortckt.insert(sortpos, i)
				currcount+=1
	p += 1
f2(sortckt)
print "Writing output file ", args.output
for k in sortckt:
	outfile.write(k[2])
outfile.close()
