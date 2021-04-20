#!/usr/bin/env python
# coding: utf-8

# # full exp:
# - gridsearch blocked/interleaved, using MSE human as eval metric
#     - fix parameters
# - eval fit on early/middle/late conditions

# # handheld splitting, RNN schema

# In[1]:


import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.stats import zscore
from utils import *




import pandas as pd
humandf = pd.read_csv('humandf.csv')
human_accD = {}
condL =['blocked','interleaved','early','middle','late']
for cond in condL:
    human_accD[cond] = humandf.loc[:,'%s mean'%cond].values
# 1d arrs of acc
human_accD['blocked'].shape,human_accD.keys()


# ### experiment funs


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

  
def calc_fit(accL,condL):
    """ 
    accL is arr [cond,seed,trials]
    NOTE: CURRENTLY FITTING ON ZSCORED
    calcualte mean squared error for model fit
    returns mse per condition
    """
    MSE = 0
    for model_acc,cond in zip(accL,condL):
        human_acc = human_accD[cond]
        # mean over seeds -> [trials]
        model_acc = model_acc.mean(0) 
        # zscore
        model_zacc = zscore(model_acc)
        human_zacc = zscore(human_acc)
        # mse
        MSE += np.sum((human_zacc - model_zacc)**2)
    return MSE


# # gridsearch free params:
# 
# - sticky_decay_rate
# - pe_thresh 
# - init_lr 
# - lr_decay_rate

# In[6]:


def paramD_to_fname(paramD):
  return "-".join(["%s_%f"%(i,j) for i,j in paramD.items()])



saving=True


nseeds_gs = 10
# gridsearch
condBI = ['blocked','interleaved']
Sd = np.arange(0.02,0.051,0.005)
Pt = np.arange(0.8,1.11,0.1)
L0 = np.arange(0.25,0.451,0.05)
Ld = np.arange(0.05,0.251,0.05)
stsizeL = [6,8,10]

print('ncond',len(Sd)*len(Pt)*len(L0)*len(Ld))



gs_results = []
for sd,pt,l0,ld,st in itertools.product(Sd,Pt,L0,Ld,stsizeL):
    paramD = {
      'sticky_decay':sd,
      'pe_thresh':pt,
      'init_lr':l0,
      'lr_decay':ld,
      'stsize':st
    }
    acc = run_exp(nseeds_gs,condBI,paramD)
    mse = calc_fit(acc,condBI)
    if saving:
      fname = paramD_to_fname(paramD)
      np.save("gsdata/"+fname,acc)
    else:
      gs_results.append({**paramD,'mse':mse})
