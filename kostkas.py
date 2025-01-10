import numpy as np
import mpnum as mp # MPS/MPO package
import random
import itertools
import time
from math import factorial

# MPS algorithm for Kostka Numbers
# computes Kostkas for a given weight vector Mu
class KostkaBuilder: 

    # Input: 
    # Mu : a list of positive integers that sums up to n. 
    def __init__(self, Mu):
        self.Mu = Mu
        self.n = np.sum(self.Mu)
        self.relerr = 0 # relative error for MPS compression
        self.tensor1 = np.zeros((1,2,1))
        self.tensor1[0,1,0] = 1 # basis state |1>
        self.tensor0 = np.zeros((1,2,1))
        self.tensor0[0,0,0] = 1 # basis state |0>
        self.get_MPS() 
        
    # Computes the Kostka K_lambda,Mu for a partition Lambda
    # Input:
    # Lambda: a non-increasing list of positive integers summing to n
    def get_kostka(self, Lambda):
        assert(len(Lambda) <= self.n)
        padded_Lambda = list(Lambda) + [0]*(self.n - len(Lambda))
        array = [self.tensor0]*(2*self.n)
        for i in range(self.n):
            array[padded_Lambda[i] + self.n -1 - i] = self.tensor1
        basis_state_mps = mp.MPArray(mp.mpstruct.LocalTensors(array))
        return mp.mparray.inner(basis_state_mps,self.mps)
    
    def num2index(self, i,  base):
        return [int(i/base), i % base]
        
    def index2num(self, index,base):
        return index[0]*base + index[1]
    
    def getMPO(self, k):
        
        array = []
        
        # index ordering LUDR
        
        # left boundary
        tensor = np.zeros((1,2,2,(k+1)**2))
        tensor[0, :, :, 0] = np.eye(2)
        tensor[0, :, :, self.index2num([1,0], k+1)] = np.array([[0, 1], [0, 0]]) # annihilate
        array.append(tensor)
        
        # bulk
        tensor = np.zeros(((k+1)**2, 2, 2, (k+1)**2))
        for i in range(k-1):
            tensor[self.index2num([i,i], k+1), :, :, self.index2num([i,i], k+1)] = np.eye(2)
            tensor[self.index2num([i,i], k+1), : , :, self.index2num([i+1, i], k+1)] = np.array([[0,1],[0,0]])
            tensor[self.index2num([i+1,i], k+1), :, :, self.index2num([i+1,i+1], k+1)] = np.array([[0,0],[1,0]])
            tensor[self.index2num([i+1, i], k+1), :, :, self.index2num([i+2, i+1],k+1)] = np.array([[1,0],[0,0]])
        
        tensor[self.index2num([k-1, k-1], k+1), :, :, self.index2num([k-1, k-1],k+1)] = np.eye(2)
        tensor[self.index2num([k,k-1], k+1), :, :, self.index2num([k, k], k+1)] = np.array([[0,0],[1,0]])
        tensor[self.index2num([k, k], k+1), :, :, self.index2num([k, k], k+1)] = np.eye(2)
        
        array = array + (2*self.n-2)*[tensor]
        
        # right boundary 
        tensor = np.zeros(((k+1)**2,2,2,1))
        tensor[self.index2num([k,k], k+1), :, :, 0] = np.eye(2)
        tensor[self.index2num([k, k-1], k+1), :, :, 0] = np.array([[0, 0],[1, 0]]) # create
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
            

n = 1

# compute all partitions of n
Pn = [list(p) for p in list(partitions(n))]
for p in Pn:
    p.reverse()
Pn = [tuple(p) for p in Pn]

print('n=',n)
print('Number of partitions=',len(Pn))

Mu = random.choice(Pn)
print('Weight Mu=',Mu)

# compute all kostas of weight Mu using the MPS algorithm
t = time.time()
builder = KostkaBuilder(Mu)
table_mps = {}
for Lambda in Pn:
    table_mps[Lambda] = builder.get_kostka(Lambda)
elapsed = time.time() - t
print('MPS runtime=',"{0:.2f}".format(elapsed))