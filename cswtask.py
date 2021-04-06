import os
import numpy as np




class Schema():

    def __init__(self):
        self.nstates = 6
        self.errD = {i:[] for i in range(self.nstates)}
        self.Tmat = self._init_transition_matrix()

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
        alfa = 0.05
        errD = self.calc_error_on_path(path)
        for st0,errvec in errD.items():
            self.Tmat[st0,:] += alfa*errvec
        return None

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
            L = []
            for s0,s1 in zip(path[:-1],path[1:]):
                L.append(self.Tmat[s0,s1])
            acc_arr.append(L)
        return np.array(acc_arr)



class Agent():

    def __init__(self):
        self.nstates = 6
        self.nschemas = 20
        self.schlib = [Schema() for i in range(self.nschemas)]
        self.errD = {i:[] for i in range(self.nstates)}
        return None 

    def select_schema(self,path,rule='nosplit'):
        if rule == 'nosplit':
            sch = self.schlib[0]
        elif rule == 'thresh':
            None
        return sch


    def forward_exp(self,exp):
        acc = []
        for tr,path in enumerate(exp): 
            self.tr = tr
            # select schema before prediction
            sch = self.select_schema(path)
            # eval
            acc.append(sch.eval())
            # update
            sch.update_sch(path)
        return np.array(acc)

    def eval_model(self,exp):
        score = np.zeros(exp.shape)
        return score



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

