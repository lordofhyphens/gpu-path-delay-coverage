import random
import argparse

parser = argparse.ArgumentParser(description="random test pattern generator")
parser.add_argument('inputs', metavar='I', type=int)
parser.add_argument('patterns', metavar='N', type=int)

args =parser.parse_args();

choices = [0,1]
z = ""
for i in range(0,args.patterns):
	for j in range(0,args.inputs):
		z += "%d"% (random.choice(choices))
	print z
	z = ""

