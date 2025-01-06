import numpy as np
np.float_ = np.float64 # for compatibility with the MPS package
np.complex_ = np.complex128 # for compatibility with the MPS package and NumPy 2.0 

import collections
collections.Sequence = collections.abc.Sequence # for compatibility with the MPS package
collections.Iterator = collections.abc.Iterator # for compatibility with the MPS package

import mpnum as mp # MPS/MPO package
import random
import itertools
import time
from functools import cache
from math import factorial



# parity of a permutation that sorts an integer sequence 

# NOTE: the cache decorator keeps the results of the function in memory, so that it does not need to be recomputed
def parity(s):
    assert(len(np.unique(s))==len(s))
    p = True
    for a,b in itertools.combinations(range(len(s)),2):
        if s[a]>s[b]:
            p = not(p)
    if p:
        return 1
    else:
        return -1

# computes the characters of the symmetric group S_n using Murnaghan–Nakayama rule (Algorithm 3 from Hepler's thesis)
#
# Inputs:
# n : defines the symmetric group S_n
# Lambda : partition of n as a list of positive integers that sums up to n
# Mu : another partition of n
#
# Returns the character of S_n for an irrep Lamnda and a conjugacy class Mu
def character(n, Lambda, Mu):
    # sort Lambda and Mu in the non-increasing order
    Lambda = np.flip(np.sort(Lambda))
    Mu = np.flip(np.sort(Mu))
    # sanity checks
    assert(np.sum(Lambda)==n)
    assert(np.sum(Mu)==n)
    assert(np.min(Lambda)>=1)
    assert(np.min(Mu)>=1)
    # number of parts in each partition
    k = len(Lambda)
    m = len(Mu)
    # extended diagram of lambda
    h_ext = [Lambda[i]+k-i-1 for i in range(k)]
    # stores the character value
    chi = 0
    # z[i] is the row of h_ext to subtract cycle Mu[i]
    z = np.zeros(m,dtype=int)
    # variable which determines which of z variables to increment next
    j = m-1
    while 1:
        continue_flag = True
        h = h_ext.copy()
        for i in range(m):
            # remove mu[i] from from row z[i] of h
            h[z[i]]-= Mu[i] 
            if h[z[i]]<0 or len(np.unique(h))<k:
                continue_flag = False
                j = i
                break

        if continue_flag: 
            chi+= parity(np.flip(np.argsort(h)))
            j = m-1

        z[j]+=1
        if z[j]<k:
            continue
        while (j>=0) and (z[j]>=(k-1)):
            z[j]=0
            j-=1
        if j<0:
            return chi
        z[j]+=1


# MPS algorithm for characters of S_n
# computes characters for a given conjugacy class Mu of S_n
class CharacterBuilder: 

    # Input: 
    # Mu : a list of positive integers that sums up to n. 
    def __init__(self, Mu):
        self.Mu = Mu
        self.n = np.sum(self.Mu)
        self.relerr = 1e-7 # relative error for MPS compression
        self.tensor1 = np.zeros((1,2,1))
        self.tensor1[0,1,0] = 1 # basis state |1>
        self.tensor0 = np.zeros((1,2,1))
        self.tensor0[0,0,0] = 1 # basis state |0>
        self.get_MPS() 

    # Computes the character chi_Lambda(Mu) for a conjugacy class Mu and an irrep Lambda of S_n
    # Input: 
    # Lambda : a list of positive integers that sums up to n. 
    def get_character(self,Lambda):
        assert(len(Lambda)<=self.n)
        padded_Lambda = list(Lambda) + [0]*(self.n - len(Lambda))
        array = [self.tensor0]*(2*self.n)
        for i in range(self.n):
            array[padded_Lambda[i] + self.n - 1 - i] = self.tensor1
        basis_state_mps = mp.MPArray(mp.mpstruct.LocalTensors(array))
        return mp.mparray.inner(basis_state_mps,self.mps)
        
    # MPO representation of the current operator J_k = sum_i a_i a_{i+k}^dag
    def getMPO(self, k):

        array = []

        # left boundary
        tensor = np.zeros((1,2,2,k+2)) 
        tensor[0, : , :, 0] = np.eye(2)
        tensor[0, :, :,  1] = np.array([[0,1],[0,0]]) # flip qubit from '1' to '0'
        array.append(tensor)

        # bulk
        # index ordering LRUD
        tensor = np.zeros((k+2, 2, 2,k+2)) 
        
        tensor[0, :, :, 0] = np.eye(2)
        tensor[k+1, :, :, k+1] = np.eye(2)
        tensor[0, :, :, 1] = np.array([[0,1],[0,0]]) # flip qubit from '1' to '0'
        tensor[k, :, :, k+1] = np.array([[0,0],[1,0]]) # flip qubit from '0' to '1'
    
        # Pauli Z 
        for j in range(1, k):
            tensor[j, :, :, j+1] = np.array([[1,0],[0,-1]])

        array = array + (2*self.n-2)*[tensor]

        # right boundary
        tensor = np.zeros((k+2, 2, 2, 1)) 
        tensor[k+1, : , : ,0] = np.eye(2)
        tensor[k, :, :, 0] = np.array([[0,0],[1,0]]) # flip qubit from '0' to '1'
        array.append(tensor)    
        return mp.MPArray(mp.mpstruct.LocalTensors(array))


    def get_MPS(self):
        # MPS representation of the initial state |1^n 0^n>
        array = self.n*[self.tensor1] + self.n*[self.tensor0]
        self.mps  = mp.MPArray(mp.mpstruct.LocalTensors(array))
        # apply a sequence of the current operators using MPO-MPS multiplication
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


n = 10

# compute all partitions of n
Pn = [list(p) for p in list(partitions(n))]
for p in Pn:
    p.reverse()
Pn = [tuple(p) for p in Pn]

print('n=',n)
print('Number of partitions=',len(Pn))

Mu = random.choice(Pn)
print('Conjugacy class Mu=',Mu)

# compute all characters of Mu using the MPS algorithm
t = time.time()
builder = CharacterBuilder(Mu)
table_mps = {}
for Lambda in Pn:
    table_mps[Lambda] = builder.get_character(Lambda)
elapsed = time.time() - t
print('MPS runtime=',"{0:.2f}".format(elapsed))

# compute all characters of Mu using Murnaghan–Nakayama rule (Hepler's algorithm)
t = time.time()
table = {}
for Lambda in Pn:
    table[Lambda] =  character(n,Lambda,Mu)
elapsed = time.time() - t
print('Murnaghan–Nakayama runtime=',"{0:.2f}".format(elapsed))

# check correctness of the MPS algorithm
err_max = 0
for Lambda in Pn:
    err_max = max(err_max, np.abs(table_mps[Lambda]-table[Lambda]))
print('maximum approximation error=',err_max)

for Lambda in Pn:
    print('Irrep=',Lambda,'character=',table[Lambda])