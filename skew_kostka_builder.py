import numpy as np
import mpnum as mp # MPS/MPO package

from utils import majorize


class SkewKostkaBuilder: 
    def __init__(self, Mu:tuple[int], Nu:tuple[int]=(0, ), relerr=1e-14):
        """
        MPS algorithm for skew Kostka numbers.

        Computes Kostkas for a given weight vector Mu and skew Nu.

        Args:
            Mu (tuple[int]): we assume that Mu is given in non-increasing order
            Nu (tuple[int], optional): _description_. Defaults to (0, ).
            relerr (_type_, optional): _description_. Defaults to 1e-14.
        """
        self.Mu = Mu
        self.n = np.sum(self.Mu)
        # m is the size of partitions such that Lambda \ Nu is valid
        self.m = self.n+np.sum(Nu)
        assert(len(Nu) <= self.n)
        assert(sum(Nu) <= self.n)
        self.Nu = Nu + [0]*(self.m-len(Nu)) # Pad Nu with 0s
        self.relerr = relerr # relative error for MPS compression
        self.tensor1 = np.zeros((1,2,1))
        self.tensor1[0,1,0] = 1 # basis state |1>
        self.tensor0 = np.zeros((1,2,1))
        self.tensor0[0,0,0] = 1 # basis state |0>
        self.get_MPS() 
        
        # divide the spin chain into four intervals: left (L), center left
        # (C1), center right C2, right (R)
        self.n1 = int(np.round(self.n / 2))
        self.n2 = self.n
        self.n3 = int(np.round(3 * self.n / 2))
        self.L = [i for i in range(2 * self.n) if i < self.n1]
        self.C1 = [i for i in range(2 * self.n)
                   if i >= self.n1 and i < self.n2]
        self.C2 = [i for i in range(2 * self.n)
                   if i >= self.n2 and i < self.n3]
        self.R = [i for i in range(2 * self.n) if i >= self.n3]
        self.nL = len(self.L)
        self.nC1 = len(self.C1)
        self.nC2 = len(self.C2)
        self.nR = len(self.R)
        # cache partial products of MPS matrices over each interval
        self.cacheL = {}
        self.cacheC1 = {}
        self.cacheC2 = {}
        self.cacheR = {}
    
    # Determines if Lamba \ Nu has enough boxes to have weight Mu
    def valid_skew(self, Lambda):
        return majorize(self.Nu, Lambda) and sum(Lambda) == self.m
        
        
    # Computes the skew Kostka K_Lambda\Nu,Mu for a partition Lambda
    # Input:
    # Lambda: a non-increasing list of positive integers summing to n
    def get_kostka(self, Lambda, valid=True):
        # check majorization condition before computing amplitudes
        if valid:
            if not self.valid_skew(Lambda):
                return 0
        
        padded_Lambda = list(Lambda) + [0]*(self.m - len(Lambda))
        if self.n < 8:
             # don't use caching for small n's
             array = [self.tensor0] * (2 * self.n)
             for i in range(self.n):
                 array[padded_Lambda[i] + self.n - 1 - i] = self.tensor1
             basis_state_mps = mp.MPArray(mp.mpstruct.LocalTensors(array))
             # compute inner product between a basis state and the MPS
             return mp.mparray.inner(basis_state_mps, self.mps)
        
        bitstring = np.zeros(2 * self.n, dtype=int)
        supp = [padded_Lambda[i] + self.n - i - 1 for i in range(self.n)]
        bitstring[supp] = 1
        # project bitstring onto each caching register
        xL = bitstring[self.L]
        xC1 = bitstring[self.C1]
        xC2 = bitstring[self.C2]
        xR = bitstring[self.R]
        
        if not (tuple(xL) in self.cacheL):
            self.cacheL[tuple(xL)] = np.linalg.multi_dot(
                [self.mps.lt[self.L[i]][:, xL[i], :] for i in range(self.nL)])

        if not (tuple(xC1) in self.cacheC1):
            self.cacheC1[tuple(xC1)] = np.linalg.multi_dot(
                [self.mps.lt[self.C1[i]][:, xC1[i], :]
                 for i in range(self.nC1)])

        if not (tuple(xC2) in self.cacheC2):
            self.cacheC2[tuple(xC2)] = np.linalg.multi_dot(
                [self.mps.lt[self.C2[i]][:, xC2[i], :]
                 for i in range(self.nC2)])

        if not (tuple(xR) in self.cacheR):
            self.cacheR[tuple(xR)] = np.linalg.multi_dot(
                [self.mps.lt[self.R[i]][:, xR[i], :]
                 for i in range(self.nR)])

        chi = (self.cacheL[tuple(xL)] @ self.cacheC1[tuple(xC1)]
               ) @ (self.cacheC2[tuple(xC2)] @ self.cacheR[tuple(xR)])
        return chi[0][0]
    
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
        
        array = array + (2*self.m-2)*[tensor]
        
        # right boundary 
        tensor = np.zeros((2*k+1,2,2,1))
        tensor[2*k, :, :, 0] = np.eye(2)
        tensor[2*k-1, :, :, 0] = np.array([[0, 0],[1, 0]]) # create
        array.append(tensor)
        
        return mp.MPArray(mp.mpstruct.LocalTensors(array))

    def get_MPS(self):
        array = [] # Local tensors
        # Traverse Nu in reverse order
        array += [self.tensor0]*self.Nu[self.m-1] # step right
        array += [self.tensor1] # step up
        for i in range(self.m-1, 0, -1):
            array += [self.tensor0]*(self.Nu[i-1]-self.Nu[i]) # step right
            array += [self.tensor1] # step up
        array = array + [self.tensor0]*(2*self.m-len(array)) # step right
        self.mps  = mp.MPArray(mp.mpstruct.LocalTensors(array))
        # apply a sequence of the h_k's using MPO-MPS multiplication
        for k in self.Mu:
            mpo = self.getMPO(k)
            self.mps = mp.dot(mpo, self.mps)
            self.mps.compress(method='svd', relerr=self.relerr)
        self.MPSready = True


