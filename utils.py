import os
import numpy as np

STSPACE_SIZE = 6 # 6 for reduced csw 




class Schema():

    def __init__(self,init_lr=0.3,lr_decay_rate=0.1):
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

    def eval(self):
        """ 
        eval schema response on all paths
        returns [npaths,nsteps] 
        where each entry is probability of correct response
        """
        task = Task()
        paths = [item for sublist in task.paths for item in sublist]
        acc_arr = []
        for path in paths:
            acc_arr.append(self.eval_path(path))
        return np.array(acc_arr)

    def eval_path(self,path):
        accL = []
        for s0,s1 in zip(path[:-1],path[1:]):
            accL.append(self.Tmat[s0,s1])
        return np.array(accL)

    def calc_pe(self,path):
        errD = self.calc_error_on_path(path)
        pe = np.sum([i**2 for i in list(errD.values())])
        return pe



class Agent():

    def __init__(self,sticky_decay_rate=0.025,pe_thresh=1,init_lr=0.3,lr_decay_rate=0.1):
        # params
        self.nstates = STSPACE_SIZE 
        self.sticky_decay_rate = sticky_decay_rate # fit
        self.pe_thresh = pe_thresh # fit
        # setup schema library
        self.sch_params = {
            'init_lr':init_lr,
            'lr_decay_rate':lr_decay_rate
        }
        self.schlib = [Schema(**self.sch_params)]
        return None 

    def select_schema(self,path,rule='thresh'):
        if self.tr==0: # edge
            return self.schlib[0]
        if rule == 'nosplit': # debug
            sch = self.schlib[0]
        elif rule == 'minpe': # debug
            sch = self.select_schema_minpe(path)
        elif rule == 'thresh': # main
            # probabilistic sticky
            if np.random.binomial(1,np.exp(-self.sticky_decay_rate*self.sch.nupdates)):
                return self.sch
            # calculate pe on active schema
            pe_sch_t = self.sch.calc_pe(path)
            # if pe below thresh: stay
            if pe_sch_t < self.pe_thresh:
                sch = self.sch
            else:
                sch = self.select_schema_minpe(path)
        return sch

    def select_schema_minpe(self,path):
        # append to schlib
        self.schlib.append(Schema(**self.sch_params))
        # 
        peL = []
        for sch in self.schlib:
            peL.append(sch.calc_pe(path))
        minpe = np.min(peL)
        return self.schlib[np.argmin(peL)]

    def forward_exp(self,exp):
        acc = []
        PE = np.zeros(len(exp))
        self.sch = self.schlib[0] 
        for tr,path in enumerate(exp): 
            PE[tr] = self.sch.calc_pe(path)
            self.tr = tr
            # update active schema
            self.sch = self.select_schema(path)
            # eval
            acc.append(self.sch.eval_path(path))
            # update
            self.sch.update_sch(path)
        return np.array(acc),PE



class Task():
    """ 
    """

    def __init__(self):
        A1,A2,B1,B2 = self._init_paths_toy()
        self.paths = [[A1,A2],[B1,B2]]
        self.tsteps = len(self.paths[0][0])
        self.exp_int = None
        return None


    def _init_paths_csw(self):
        """ 
        begin -> locA -> node11, node 21, node 31, end
        begin -> locA -> node12, node 22, node 32, end
        begin -> locB -> node11, node 22, node 31, end
        begin -> locB -> node12, node 21, node 32, end
        """
        locA,locB = 0,1
        node11,node12 = 3,4
        node21,node22 = 5,6
        A1 = np.array([locA,
            node11,node21,end
            ])
        A2 = np.array([locA,
            node12,node22,end
            ])
        B1 = np.array([locB,
            node11,node22,end
            ])
        B2 = np.array([locB,
            node12,node21,end
            ])
        return A1,A2,B1,B2

    def _init_paths_toy(self):
        """ 
        begin -> locA -> node11, node 21, node 31, end
        begin -> locA -> node12, node 22, node 32, end
        begin -> locB -> node11, node 22, node 31, end
        begin -> locB -> node12, node 21, node 32, end
        """
        locA,locB = 0,1
        node11,node12 = 2,3
        node21,node22 = 4,5
        A1 = np.array([locA,
            node11,node21
            ])
        A2 = np.array([locA,
            node12,node22
            ])
        B1 = np.array([locB,
            node11,node22
            ])
        B2 = np.array([locB,
            node12,node21
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

