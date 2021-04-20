import sys
import itertools

"""
controls the gridsearch parameter space
given an index, return parameter string
""" 

param_set_idx = int(sys.argv[1])

Sd = np.arange(0.02,0.051,0.005)
Pt = np.arange(0.8,1.11,0.1)
L0 = np.arange(0.25,0.451,0.05)
Ld = np.arange(0.05,0.251,0.05)
stsizeL = [5,6,7]

print('nconds',len(Sd)*len(Pt)*len(L0)*len(Ld))


itrprod = itertools.product(Sd,Pt,L0,Ld,stsizeL)

for idx,paramL in enumerate(itrprod):
  if idx == param_set_idx:
    print(" ".join([str(i) for i in paramL]))
    break


