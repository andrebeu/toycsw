import os
import numpy as np


class Agent():

    def __init__(self):
        self.num_states = 6
        self.Tmat = self.init_transition_matrix()
        self.errD = {i:[] for i in range(self.num_states)}
        return None

    def init_transition_matrix(self):
        T = np.random.random((self.num_states,self.num_states))
        # rows sum to one
        T = np.transpose(T/T.sum(axis=0))
        return T

    def forward_path(self,path):
        # print(np.round(self.T,2))
        alfa = 0.2
        for st,stp1 in zip(path[:-1],path[1:]):
            # print(st)
            # calculate error
            obs = np.zeros(self.num_states)
            obs[stp1] = 1
            # print(st)
            error = obs-self.Tmat[st]
            # record error
            self.Emat[self.tr,st] += np.sum(error**2)
            self.errD[st].append(np.sum(error**2))
            # update transition matrix
            self.Tmat[st,:] += alfa*error
        # assert False
        return error

    def forward_exp(self,exp):
        self.Emat = np.zeros((len(exp),self.num_states))
        for tr,path in enumerate(exp): 
            self.tr = tr
            error_tr = self.forward_path(path)
        return self.Emat



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

