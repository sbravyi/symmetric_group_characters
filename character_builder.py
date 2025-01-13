import numpy as np
import mpnum as mp # MPS/MPO simulation package


# MPS algorithm for characters of the symmetric group S_n described in arXiv:2501.?????
class CharacterBuilder: 

    # Takes as input a conjugacy class Mu of S_n specified as a list of positive integers that sum to n
    def __init__(self, Mu):
        self.Mu = list(np.sort(Mu))
        self.n = np.sum(self.Mu)
        self.relerr = 1e-10 # relative error for MPS compression
        self.tensor1 = np.zeros((1,2,1))
        self.tensor1[0,1,0] = 1 # basis state |1>
        self.tensor0 = np.zeros((1,2,1))
        self.tensor0[0,0,0] = 1 # basis state |0>
        self.maximum_rank = 1 # maximum MPS bond dimension (maximum Schmidt rank)
        # compute the MPS that encodes all characters of Mu
        self.get_MPS()
       

        # divide the spin chain into four intervals: left (L), center left (C1), center right C2, right (R)
        self.n1 = int(np.round(self.n/2))
        self.n2 = self.n
        self.n3 = int(np.round(3*self.n/2))
        self.L = [i for i in range(2*self.n) if i<self.n1]
        self.C1 = [i for i in range(2*self.n) if i>=self.n1 and i<self.n2]
        self.C2 = [i for i in range(2*self.n) if i>=self.n2 and i<self.n3]
        self.R = [i for i in range(2*self.n) if i>=self.n3]
        self.nL = len(self.L)
        self.nC1 = len(self.C1)
        self.nC2 = len(self.C2)
        self.nR = len(self.R)
        # cache partial products of MPS matrices over each interval
        self.cacheL = {}
        self.cacheC1 = {}
        self.cacheC2 = {}
        self.cacheR = {}
      

    # Computes the character chi_Lambda(Mu) for a conjugacy class Mu and an irrep Lambda of S_n
    # Input: 
    # Lambda : a list of positive integers that sums up to n. 
    def get_character(self,Lambda):
        assert(len(Lambda)<=self.n)
        # pad the partition Lambda with zeros to make n parts
        padded_Lambda = list(Lambda) + [0]*(self.n - len(Lambda))
        if self.n<8:
            # don't use caching for small n's
            array = [self.tensor0]*(2*self.n)
            for i in range(self.n):
                array[padded_Lambda[i] + self.n - 1 - i] = self.tensor1
            basis_state_mps = mp.MPArray(mp.mpstruct.LocalTensors(array))
            # compute inner product between a basis state and the MPS
            return mp.mparray.inner(basis_state_mps,self.mps)

        bitstring = np.zeros(2*self.n,dtype=int)
        supp = [padded_Lambda[i] + self.n - i - 1 for i in range(self.n)]
        bitstring[supp] = 1
        # project bitstring onto each caching register
        xL = bitstring[self.L]
        xC1 = bitstring[self.C1]
        xC2 = bitstring[self.C2]
        xR = bitstring[self.R]

        if not(tuple(xL) in self.cacheL):
            self.cacheL[tuple(xL)] = np.linalg.multi_dot([self.mps.lt[self.L[i]][:,xL[i],:] for i in range(self.nL)])
        
        if not(tuple(xC1) in self.cacheC1):
            self.cacheC1[tuple(xC1)] = np.linalg.multi_dot([self.mps.lt[self.C1[i]][:,xC1[i],:] for i in range(self.nC1)])

        if not(tuple(xC2) in self.cacheC2):
            self.cacheC2[tuple(xC2)] = np.linalg.multi_dot([self.mps.lt[self.C2[i]][:,xC2[i],:] for i in range(self.nC2)])

        if not(tuple(xR) in self.cacheR):
            self.cacheR[tuple(xR)] = np.linalg.multi_dot([self.mps.lt[self.R[i]][:,xR[i],:] for i in range(self.nR)])


        chi = (self.cacheL[tuple(xL)] @ self.cacheC1[tuple(xC1)]) @ (self.cacheC2[tuple(xC2)] @ self.cacheR[tuple(xR)])
        return chi[0][0]

    # MPO representation of the current operator J_k = sum_i a_i a_{i+k}^dag
    def getMPO(self, k):

        array = []

        # left boundary
        tensor = np.zeros((1,2,2,k+2)) 
        tensor[0, : , :, 0] = np.eye(2)
        tensor[0, :, :,  1] = np.array([[0,1],[0,0]]) # flip qubit from '1' to '0'
        array.append(tensor)

        # bulk
        # index ordering Left Right Up Down
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
        self.maximum_rank = 1
        for k in self.Mu:
            mpo = self.getMPO(k)
            self.mps = mp.dot(mpo, self.mps)
            self.mps.compress(method='svd', relerr=self.relerr)
            self.maximum_rank = max( self.maximum_rank,np.max(self.mps.ranks))


    def get_bond_dimension(self):
        return self.maximum_rank
        


# generates all partitions of n 
# source: https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
def partitions(n, I=1):
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p

def get_partitions(n):
    Pn = [list(p) for p in list(partitions(n))]
    for p in Pn:
        p.reverse()
    return [tuple(p) for p in Pn]
