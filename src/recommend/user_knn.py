'''
user_knn.py

Class instance of userKNN model from GroupLens paper.
'''
import random
import numpy as np
import pandas as pd

class UserKNN():
    def __init__(self, A, row_idx=None, col_idx=None):
        '''
        Create instance of UserKNN model
        Attributes:
            A: sparse nxp matrix where there are n items and p reviewers. Empty cells 
                are represented by np.nan.
            n, p: dimensions of A.
            M: mask matrix of A where invalid cells are False and valid cells are True.
            mu: the average rating of each reviewer/column, size 1xp.
            R: correlation coefficient matrix of size pxp, with diagonals set to 0. 
            P: matrix of predictions, size nxp.
        
        Methods:
            gen_valid_idx: create list of tuple (i, j) that index cells with entries
            train_test_split: TODO
            gen_corrcoef: TODO
            gen_preds: TODO
        '''
        self.A = A
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.valid_idx = None
        self.test_idx = None
        self.A_prime = None
        self.n, self.p = A.shape
        self.M = None
        self.mu = None 
        self.R = None
        self.P = None

    def prune_A(self):
        '''
        Prune A to min number of reviews in column or row 
        NOTE: maybe this will happen before loading, so no need for this method
            Probably better to do ahead of time since multiple models will use
        '''
        #row_sum = np.sum(~np.isnan(A), axis = 1)
        #col_sum = np.sum(~)
        pass

    def gen_valid_idx(self):
        valid_i, valid_j = np.where(~np.isnan(self.A))
        self.valid_idx = list(zip(valid_i, valid_j))
    
    def train_test_split(self, s=37, test_size=0.2):
        '''
        Split A into train and test data, including omission mask for test

        Inputs:
            test_size: float between 0 and 1 indicating the fraction of valid entries
                to be reserved for testing
        
        Outputs:
            A list of indices to omit from self.valid_idx
        '''
        random.seed(s)
        # how many
        n_entries = len(self.valid_idx)
        test_n = round(n_entries * test_size)
        test_entry_idx = random.sample(range(n_entries), test_n) 
        self.test_idx = [self.valid_idx[i] for i in test_entry_idx]
        self.A_prime = self.A.copy()
        for t in self.test_idx:
            i, j = t
            self.A_prime[i, j] = np.nan
    
    def gen_M(self):
        self.M = ~np.ma.masked_invalid(self.A_prime).mask

    def gen_mu(self):
        self.mu = np.nansum(self.A_prime, axis=0) / np.sum(self.M, axis=0)

    def gen_corrcoef(self):
        # Numpy version of corrcoef with NaN's gives abs. values greater than 1
        # perhaps because np.ma.corrcoef doesn't work properly
        # see https://github.com/numpy/numpy/issues/15601
        # self.R = np.ma.corrcoef(np.ma.masked_invalid(self.A), rowvar=False).data
        self.R = pd.DataFrame(self.A_prime).corr().to_numpy() # use Pandas df.corr() instead
        np.fill_diagonal(self.R, 0)

    def gen_preds(self):
        '''
        For each reviewer-album, predict a rating, returning an nxp array P

        '''
        A_zeros = np.nan_to_num(self.A_prime, copy=True, nan=0)
        
        R_zeros = np.nan_to_num(self.R, copy=True, nan=.01)
        D = self.M @ abs(R_zeros)

        D0 = np.ones([self.n, self.n])
        np.fill_diagonal(D0, 0)

        #print(f'---A zeroes---')
        #print(f'{A_zeros}')
        #print(f'---R zeroes---')
        #print(f'{R_zeros}')
        J_bar = (D0 @ A_zeros) / (D0 @ self.M)
        #print(f'---J_Bar---')
        #print(f'{J_bar}')
        #print(f'D {D}')
        #print(f'A - J_bar {self.A - J_bar}')
        self.P = self.mu + (np.nan_to_num(self.A_prime - J_bar, nan=0) @ R_zeros) / D
        


    def eval_performance(self):
        '''
        Calculate performace
        '''
        # TODO 
        pass
    
    def combine_preds_original(self):
        # TODO
        pass 