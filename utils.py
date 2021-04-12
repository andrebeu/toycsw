import os
import numpy as np
import torch as tr

STSPACE_SIZE = 10 

""" 
missing softmax prediction projection
"""


class RNNSch(tr.nn.Module):

    def __init__(self,init_lr,lr_decay_rate):
        super().__init__()
        self.stsize = 6
        self.obsdim = STSPACE_SIZE 
        self._setup()
        self.init_lr = init_lr
        self.lr_decay_rate = lr_decay_rate
        self.nupdates = 1 # used for decay
        return None

    def _setup(self):
        """ build and setup optimizer
        """
        # build
        self.embed_st = tr.nn.Embedding(self.obsdim,self.stsize)
        self.rnn = tr.nn.GRU(self.stsize,self.stsize)
        self.init_rnn = tr.nn.Parameter(tr.rand(2,1,1,self.stsize),requires_grad=True)
        self.ffout = tr.nn.Linear(self.stsize,self.obsdim,bias=False)
        # optimizer setup
        self.lossop = tr.nn.CrossEntropyLoss()

    def get_lr(self):
        return self.init_lr*np.exp(-self.lr_decay_rate*self.nupdates)

    def forward(self,path): 
        ''' path -> 1D[obs1,obs2...] 
        return yhat -> arr[tstep,batch,feature]
        '''
        # force type, include batch dim
        x_t = tr.tensor(path).unsqueeze(1) 
        # first projection
        x_t = self.embed_st(x_t)
        # rnn
        h0,_ = self.init_rnn
        yhat,hn = self.rnn(x_t,h0)
        # output layer
        yhat = self.ffout(yhat)
        return yhat

    def update(self,path):
        # setup
        lr = self.get_lr()
        self.optiop = tr.optim.Adam(self.parameters(), lr=lr)
        # fwprop all but first, target starts from second obs
        yhat = self.forward(path[:-1])
        ytarget = tr.tensor(path[1:]).unsqueeze(1)
        ## ??implicit loop backprop through time??
        # ytarget = tr.tensor(path)
        # loss = self.lossop(yhat.squeeze(),ytarget)
        ## explicit loop
        self.optiop.zero_grad()
        loss = 0
        for yh_t,yt_t in zip(yhat,ytarget):
            loss += self.lossop(yh_t,yt_t)
            loss.backward(retain_graph=True)
        self.optiop.step()
        ### update count (used for lr decay)
        self.nupdates += 1
        return loss

    def eval(self,path):
        """ return acc -> np.1D[tsteps]
        """
        yh = self.forward(path[:-1])
        yhsm = tr.softmax(yh,-1).squeeze()
        # 1D sm activation of target 
        yhsm_target = yhsm[range(len(yhsm)),path[1:]]
        # 
        acc = yhsm_target.detach().numpy()
        return acc


class TmatSch():

    def __init__(self,init_lr,lr_decay_rate):
        self.nstates = STSPACE_SIZE
        # paramS
        self.init_lr = init_lr  # fit
        self.lr_decay_rate = lr_decay_rate # fit; larger faster decay 
        # init objects
        self.Tmat = self._init_transition_matrix()
        self.nupdates = 1

    def _init_transition_matrix(self):
        # T[s0,s1] = pr(s1|s0)
        T = np.random.random((self.nstates,self.nstates))
        T = np.transpose(T/T.sum(axis=0)) # rows sum to one
        return T

    def calc_error_obs(self,st0,st1):
        """ delta err vec is len Nstates
        O(st0) - pr(st1), where pr(st1) 
        is next state prediciton (softmax)
        """
        obs = np.zeros(self.nstates)
        obs[st1] = 1
        delta_err_vec = obs-self.Tmat[st0]
        return delta_err_vec

    def calc_error_on_path(self,path):
        """ returns {st0: delta_err_vec} for st0 in path
        """
        D = {}
        for st0,st1 in zip(path[:-1],path[1:]):
            D[st0] = self.calc_error_obs(st0,st1)
        return D

    def update_sch(self,path):
        lr=self.get_lr()
        errD = self.calc_error_on_path(path)
        for st0,errvec in errD.items():
            self.Tmat[st0,:] += lr*errvec
        self.nupdates += 1
        return None

    def get_lr(self):
        return self.init_lr*np.exp(-self.lr_decay_rate*self.nupdates)

    def eval_path(self,path):
        """ 
        return acc > 1D len(tsteps)
        """
        accL = []
        for s0,s1 in zip(path[:-1],path[1:]):
            accL.append(self.Tmat[s0,s1])
        return np.array(accL)

    def calc_pe(self,path):
        errD = self.calc_error_on_path(path)
        pe = np.sum([i**2 for i in list(errD.values())])
        return pe





class Agent():

    def __init__(self,sticky_decay_rate=0.02,pe_thresh=1,init_lr=0.25,lr_decay_rate=0.05):
        # params
        self.nstates = STSPACE_SIZE 
        # fitting params
        self.sticky_decay_rate = sticky_decay_rate 
        self.pe_thresh = pe_thresh 
        # NB sch params common across delta & RNN
        self.sch_params = { 
            'init_lr':init_lr,
            'lr_decay_rate':lr_decay_rate
        }
        # setup schema library
        self.schlib = [{
            'tsch':TmatSch(**self.sch_params),
            'rsch':RNNSch(**self.sch_params)
            }]
        return None 

    def select_schema(self,path,rule='thresh'):
        """ refactor to return schema index 
        """
        if self.tr==0: # edge
            return self.schlib[0]
        if rule == 'nosplit': # debug
            sch = self.schlib[0]
        elif rule == 'thresh': # main
            # probabilistic sticky
            pr_stay = np.exp(-self.sticky_decay_rate*self.sch['tsch'].nupdates)
            stay = np.random.binomial(1,pr_stay)
            if stay:
                return self.sch
            # calculate pe on active schema
            pe_sch_t = self.sch['tsch'].calc_pe(path)
            # if pe below thresh: stay
            if pe_sch_t < self.pe_thresh:
                sch = self.sch
            else:
                sch = self._select_schema_minpe(path)
        return sch

    def _select_schema_minpe(self,path):
        # append to schlib
        self.schlib.append({
            'tsch':TmatSch(**self.sch_params),
            'rsch':RNNSch(**self.sch_params)
            })
        # 
        peL = []
        for sch in self.schlib:
            peL.append(sch['tsch'].calc_pe(path))
        minpe = np.min(peL)
        return self.schlib[np.argmin(peL)]

    def forward_exp(self,exp):
        """ exp -> arr[trials,tsteps]
        acc > [ntrils,tsteps]
        """
        accT = []
        accR = []
        self.sch = self.schlib[0] 
        for tr,path in enumerate(exp): 
            self.tr = tr
            # update active schema
            self.sch = self.select_schema(path)
            # eval
            accT.append(self.sch['tsch'].eval_path(path))
            accR.append(self.sch['rsch'].eval(path))
            # update
            self.sch['tsch'].update_sch(path)
            self.sch['rsch'].update(path)
        return np.array(accT),np.array(accR)



class Task():
    """ 
    """

    def __init__(self):
        A1,A2,B1,B2 = self._init_paths()
        self.paths = [[A1,A2],[B1,B2]]
        self.tsteps = len(self.paths[0][0])
        self.exp_int = None
        return None


    def _init_paths(self):
        """ 
        begin -> locA -> node11, node 21, node 31, end
        begin -> locA -> node12, node 22, node 32, end
        begin -> locB -> node11, node 22, node 31, end
        begin -> locB -> node12, node 21, node 32, end
        """
        begin,locA,locB = 0,1,2
        node11,node12 = 3,4
        node21,node22 = 5,6
        node31,node32 = 7,8
        end = 9
        A1 = np.array([begin,locA,
            node11,node21,node31,end
            ])
        A2 = np.array([begin,locA,
            node12,node22,node32,end
            ])
        B1 = np.array([begin,locB,
            node11,node22,node31,end
            ])
        B2 = np.array([begin,locB,
            node12,node21,node32,end
            ])
        return A1,A2,B1,B2


    def get_curriculum(self,condition,n_train,n_test):
        """ 
        order of events
        NB blocked: ntrain needs to be divisible by 4
        """
        curriculum = []   
        if condition == 'blocked':
            assert n_train%4==0
            curriculum =  \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4 )
        elif condition == 'early':
            curriculum =  \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0, 1] * (n_train // 4)
        elif condition == 'middle':
            curriculum =  \
                [0, 1] * (n_train // 8) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0, 1] * (n_train // 8)
        elif condition == 'late':
            curriculum =  \
                [0, 1] * (n_train // 4) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4)
        elif condition == 'interleaved':
            curriculum = [0, 1] * (n_train // 2)
        elif condition == 'single': ## DEBUG
            curriculum =  \
                [0] * (n_train) 
        else:
            print('condition not properly specified')
            assert False
        # 
        curriculum += [int(np.random.rand() < 0.5) for _ in range(n_test)]
        return np.array(curriculum)


    def generate_experiment(self,condition,n_train,n_test):
        """ 
        exp: arr [ntrials,tsteps]
        curr: arr [ntrials]
        """
        # get curriculum
        n_trials = n_train+n_test
        curr = self.get_curriculum(condition,n_train,n_test)
        # generate trials
        exp = -np.ones([n_trials,self.tsteps],dtype=int)
        for trial_idx in range(n_train+n_test):
            # select A1,A2,B1,B2
            event_type = curr[trial_idx]
            path_type = np.random.randint(2)
            path_int = self.paths[event_type][path_type]
            # embed
            exp[trial_idx] = path_int
        return exp,curr

