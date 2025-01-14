import numpy as np
import mpnum as mp # MPS/MPO package
import random as rm
import time
from sage.all import * # SageMath package
from functools import cache 

# MPS algorithm for Kostka Numbers
# computes Kostkas for a given weight vector Mu
# we assume that Mu is given in non-increasing order
class KostkaBuilder: 

    # Input: 
    # Mu : a list of positive integers that sums up to n. 
    def __init__(self, Mu, relerr=1e-10):
        self.Mu = Mu
        self.n = np.sum(self.Mu)
        self.relerr = relerr # relative error for MPS compression
        self.tensor1 = np.zeros((1,2,1))
        self.tensor1[0,1,0] = 1 # basis state |1>
        self.tensor0 = np.zeros((1,2,1))
        self.tensor0[0,0,0] = 1 # basis state |0>
        self.get_MPS() 
    
    # Determines if lambda >= Mu in majorization order
    # Input:
    # Lambda: a list of non-increasing positive integers summing to n
    def majorize(self, Lambda):
        sum_mu = 0
        sum_lm = 0
        
        for i in range(min(len(Lambda), len(self.Mu))):
            sum_mu += self.Mu[i]
            sum_lm += Lambda[i]
            if sum_mu > sum_lm:
                return False
        if np.sum(self.Mu) == np.sum(Lambda):
            return True
        else:
            return False
        
    # Computes the Kostka K_lambda,Mu for a partition Lambda
    # Input:
    # Lambda: a non-increasing list of positive integers summing to n
    def get_kostka(self, Lambda, maj=True):
        assert(len(Lambda) <= self.n)
        # check majorization condition before computing amplitudes
        if maj:
            if not self.majorize(Lambda):
                return 0
        
        padded_Lambda = list(Lambda) + [0]*(self.n - len(Lambda))
        array = [self.tensor0]*(2*self.n)
        for i in range(self.n):
            array[padded_Lambda[i] + self.n -1 - i] = self.tensor1
        basis_state_mps = mp.MPArray(mp.mpstruct.LocalTensors(array))
        return mp.mparray.inner(basis_state_mps,self.mps)
    
    # Converts i in range [base**2] to a pair of indices [a,b]
    @cache
    def num2index(self, i,  base):
        return [int(i/base), i % base]
    
    # Converts a pair of indices [a,b] to an integer in base "base"
    @cache
    def index2num(self, index,base):
        return index[0]*base + index[1]
    
    # Returns a MPO representing (operator) complete symmetric polynomials
    def getMPO(self, k):
        
        array = []
        
        # index ordering LUDR

        # left boundary
        tensor = np.zeros((1,2,2,2*k+1))
        tensor[0, :, :, 0] = np.eye(2)
        tensor[0, :, :, 1] = np.array([[0,1], [0,0]]) # annihilate
        array.append(tensor)
        
        # bulk
        tensor = np.zeros((2*k+1, 2, 2, 2*k+1))
        for i in range(k-1): # runs until k-2
            tensor[2*i , :, : , 2*i] = np.eye(2)
            tensor[2*i+1, :, :, 2*i+2] = np.array([[0,0],[1,0]])
            tensor[2*i+1, :, :, 2*i+3] = np.array([[1,0],[0,0]])
            tensor[2*i, :, :, 2*i+1] = np.array([[0,1],[0,0]])
            
        tensor[2*k-2, :, :, 2*k-2] = np.eye(2)
        tensor[2*k-2, :, :, 2*k-1] = np.array([[0,1],[0,0]])
        tensor[2*k-1, :, :, 2*k] = np.array([[0,0], [1,0]])
        tensor[2*k, :, :, 2*k] = np.eye(2)
        
        array = array + (2*self.n-2)*[tensor]
        
        # right boundary 
        tensor = np.zeros((2*k+1,2,2,1))
        tensor[2*k, :, :, 0] = np.eye(2)
        tensor[2*k-1, :, :, 0] = np.array([[0, 0],[1, 0]]) # create
        array.append(tensor)
        
        return mp.MPArray(mp.mpstruct.LocalTensors(array))

    def get_MPS(self):
        # MPS representation of the initial state |1^n 0^n>
        array = self.n*[self.tensor1] + self.n*[self.tensor0]
        self.mps  = mp.MPArray(mp.mpstruct.LocalTensors(array))
        # apply a sequence of the h_k's using MPO-MPS multiplication
        for k in self.Mu:
            mpo = self.getMPO(k)
            self.mps = mp.dot(mpo, self.mps)
            self.mps.compress(method='svd', relerr=self.relerr)
        self.MPSready = True


# generates all partitions of n 
# source: https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
def partitions(n, I=1):
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p
            