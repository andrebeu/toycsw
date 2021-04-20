#!/usr/bin/env python
# coding: utf-8

""" 
run single param
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.stats import zscore
from utils import *
import sys

# parameter input as string
paramstr = str(sys.argv[1])
sd,pt,l0,ld,st = paramstr.split()
paramD = {
  'sticky_decay':float(sd),
  'pe_thresh':float(pt),
  'init_lr':float(l0),
  'lr_decay':float(ld),
  'stsize':int(st)
}

nseeds_gs = 1


def run_exp(nseeds,condL,paramD,ntr=160,nte=40):
    """ returns acc for cond in condL """
    print('N=%i'%nseeds,paramD)
    acc = -np.ones([len(condL),nseeds,ntr+nte])
    for ci,cond in enumerate(condL):
        for s in range(nseeds):
            # seed ctrl
            np.random.seed(s)
            tr.manual_seed(s)
            # init
            ag = Agent(**paramD)
            task = Task()
            # run
            exp,cur = task.generate_experiment(cond,ntr,nte)
            acc[ci,s] = ag.forward_exp(exp) 
    return acc


def paramD_to_fname(paramD):
  return "-".join(["%s_%.4f"%(i,j) for i,j in paramD.items()])


# run
condBI = ['blocked','interleaved']
acc = run_exp(nseeds_gs,condBI,paramD)
# save
fname = paramD_to_fname(paramD)
np.save("gsdata/gs1/accBI-"+fname,acc)

print('done')