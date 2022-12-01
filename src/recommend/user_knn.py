'''
user_knn.py

Class instance of userKNN model from GroupLens paper.
'''
import random
import json, argparse
import numpy as np
import pandas as pd

class UserKNN:
    def __init__(self, A, row_idx=None, col_idx=None):
        '''
        Create instance of UserKNN model. Evalutation setup can be done with weak 
        generalization (train/test are disjoint on user-item, but not user) or
        strong generalization (train/test are disjoint on user). The default is weak. 
        
        Inputs:
            A: sparse nxp matrix where there are n items and p users. Empty cells 
                are represented by np.nan.
            row_idx: list of item ids in same order as rows of A
            col_idx: list of user id's in same order as columns of A

        Attributes:
            A: sparse nxp matrix where there are n items and p users. Empty cells 
                are represented by np.nan.
            n, p: dimensions of A.
            M: mask matrix of A where invalid cells are False and valid cells are True.
            mu: the average rating of each reviewer/column, size 1xp.
            R: correlation coefficient matrix of size pxp, with diagonals set to 0. 
            P: matrix of predictions, size nxp.
        
        Methods:
            gen_valid_idx: create list of tuple (i, j) that index cells with entries
            split_test_weak: TODO
            gen_corrcoef: TODO
            gen_preds: TODO
        '''
        self.A = A
        self.B = None # TODO - create function to derive this?
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.A_valid_idx = None
        self.A_test_idx = None
        self.b_valid_idx = None # TODO - create function to derive this
        self.A_prime = None
        self.n, self.p = A.shape
        self.M = None
        self.M_strong = None
        self.mu = None 
        self.mu_strong = None
        self.R = None
        self.R_strong = None
        self.P = None
        self.P_strong = None

    def gen_valid_idx(self, strong=False):
        if strong:
            V = self.B
        else:
            V = self.A
        
        valid_i, valid_j = np.where(~np.isnan(V))
        
        if strong:
            self.B_valid_idx = list(zip(valid_i, valid_j))
        else:
            self.A_valid_idx = list(zip(valid_i, valid_j))
    
    def split_test_weak(self, s=37, test_size=0.2):
        '''
        Split A into train and test data, including omission mask for test

        Inputs:
            test_size: float between 0 and 1 indicating the fraction of valid entries
                to be reserved for testing
        
        Outputs:
            A list of indices to omit from self.A_valid_idx
        '''
        random.seed(s)
        n_entries = len(self.A_valid_idx)
        test_n = round(n_entries * test_size)
        test_entry_idx = random.sample(range(n_entries), test_n) 
        self.A_test_idx = [self.A_valid_idx[i] for i in test_entry_idx]
        self.A_prime = self.A.copy()
        for t in self.A_test_idx:
            i, j = t
            self.A_prime[i, j] = np.nan
    
    def gen_M(self, strong=False):
        if strong:
            self.M_strong = ~np.ma.masked_invalid(self.A).mask
        else:
            self.M = ~np.ma.masked_invalid(self.A_prime).mask

    def gen_mu(self, strong=False):
        if strong:
            self.mu_strong = np.nanmean(self.B, axis=0)
        else:
            self.mu = np.nanmean(self.A_prime, axis=0)

    def gen_corrcoef(self, strong=False):
        if strong:
            A = pd.DataFrame(self.A)
            B = pd.DataFrame(self.B)
            self.R_strong = B.apply(lambda s: A.corrwith(s)).to_numpy()
        else:
            self.R = pd.DataFrame(self.A_prime).corr().to_numpy()
            np.fill_diagonal(self.R, 0)

    def gen_preds(self, strong=False):
        '''
        For each reviewer-album, predict a rating, returning an nxp (weak) or
        nxq (strong) array P
        '''
        if strong:
            A = self.A
            A_zeros = np.nan_to_num(self.A, copy=True, nan=0)
            R = self.R_strong
            mu = self.mu_strong
            M = self.M_strong
        else:
            A = self.A_prime
            A_zeros = np.nan_to_num(self.A_prime, copy=True, nan=0)
            R = self.R
            mu = self.mu
            M = self.M

        R_zeros = np.nan_to_num(R, copy=True, nan=.01)
        D = M @ abs(R_zeros)
        D0 = np.ones([self.n, self.n])
        np.fill_diagonal(D0, 0)

        J_bar = (D0 @ A_zeros) / (D0 @ M)

        P = mu + (np.nan_to_num(A - J_bar, nan=0) @ R_zeros) / D
        if strong:
            self.P_strong = [P[i,j] for j, i in enumerate(self.b_valid_idx)]
        else:
            self.P = P 
        
        #TEST SCRATCH

        #print(f'---A zeroes---')
        #print(f'{A_zeros}')
        #print(f'---R zeroes---')
        #print(f'{R_zeros}')
        #print(f'---J_Bar---')
        #print(f'{J_bar}')
        #print(f'D {D}')
        #print(f'A - J_bar {self.A - J_bar}')

    def eval_performance(self):
        '''
        Calculate performace
        '''
        # TODO 
        pass
    
    def combine_preds_original(self):
        # TODO
        pass 