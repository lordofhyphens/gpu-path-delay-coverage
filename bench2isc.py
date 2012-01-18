import argparse
import re
def toint(z):
	return int(z)
def getfanins(lst, d):
	out = ""
	for g in lst:
		if d[g][2] > 1:
			d[g][3]+= 1
		out += "%d " % (d[g][3])
	return out	

parser = argparse.ArgumentParser(description='Convert ISCAS89 to ISCAS85')
parser.add_argument('benchmark', metavar='IN', type=str,
                   help='input')
args = parser.parse_args()

f = open(args.benchmark, 'r')
pis = list()
pos = list()
fins = dict()
counter = 1

for j in f:
	j = re.sub("\n","",j)
	r = re.split("#",j,2)
	j = r[0]
	if len(r) > 1:
		comment = r[1]
	else:
		comment = "";
	if len(r[0]) > 1:
		if re.search("INPUT", j): 
			j = re.sub("INPUT\(","",j)
			j = re.sub("\)","",j)
			fins[j] = ['inpt', [], 0, counter]
			counter +=1 
			continue;
		elif re.search("OUTPUT", j):
			j = re.sub("OUTPUT\(","",j)
			j = re.sub("\)","",j)
			pos.append(j);
			continue;
		if re.search("=",j):
			ar = re.split("=", j, 2)
			out = re.sub(" ", "", ar[0]);
			ar = ar[1]
			ar = re.split("\(",ar,2)
			fins[out] = [re.sub(" ", "", ar[0]).lower(),re.split(",", re.sub("[() ]","",ar[1])), 0, counter]
			counter += 1
for i in fins.keys():
	for r in fins:
		if i in fins[r][1]:
			fins[i][2]+= 1
for i in pos:
	fins[i][2] += 1
counter = 1
#print fins
for i in sorted(fins.iterkeys(), cmp=lambda x,y: fins[x][3] <= fins[y][3]):
	if fins[i][0] == "inpt": 
		print "%d %s inpt %d 0 >sa0" % (counter, i, fins[i][2])
		fins[i][3] = counter
	else:
		print "%d %s %s %d %d >sa0" % (counter, i, fins[i][0], fins[i][2], len(fins[i][1]))
		fins[i][3] = counter
		lst = getfanins(fins[i][1],fins)
		print "     %s" % (lst)
	counter += 1
	if fins[i][2] > 1:
		for z in range(0,fins[i][2]):
			print "%d %sfan from %s >sa0" % (counter, i, i)
			counter += 1
