import numpy as np
import quimb.tensor as qtn 
from quimb.tensor.tensor_1d_compress import mps_gate_with_mpo_direct

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
        self.qubits = [i for i in range(2*self.n)]
        # compute the MPS that encodes all characters of Mu
        self.get_MPS()
       

    def get_character(self,Lambda):
        assert(len(Lambda)<=self.n)
        padded_Lambda = list(Lambda) + [0]*(self.n - len(Lambda))
        bit_string = np.zeros(2*self.n,dtype=int)
        supp = [padded_Lambda[i] + self.n - 1 - i for i in range(self.n)]
        bit_string[supp] = 1
        return self.mps.amplitude(bit_string)
       
      
    # MPO representation of the current operator J_k = sum_i a_i a_{i+k}^dag
    def getMPO(self, k):

        array = []

        # left boundary
        tensor = np.zeros((k+2,2,2)) 
        tensor[0, : , :] = np.eye(2)
        tensor[1, : , :] = np.array([[0,1],[0,0]]) # flip qubit from '1' to '0'
        array.append(tensor)

        # bulk
        tensor = np.zeros((k+2,k+2, 2, 2)) 
        tensor[0, 0, :, :] = np.eye(2)
        tensor[k+1,k+1, :, :] = np.eye(2)
        tensor[0, 1, :, :] = np.array([[0,1],[0,0]]) # flip qubit from '1' to '0'
        tensor[k, k+1, :, :] = np.array([[0,0],[1,0]]) # flip qubit from '0' to '1'
    
        # Pauli Z 
        for j in range(1, k):
            tensor[j, j+1, :, :] = np.array([[1,0],[0,-1]])

        array = array + (2*self.n-2)*[tensor]

        # right boundary
        tensor = np.zeros((k+2, 2, 2)) 
        tensor[k+1, : , : ] = np.eye(2)
        tensor[k, :, :] = np.array([[0,0],[1,0]]) # flip qubit from '0' to '1'
        array.append(tensor)      
    
        return qtn.tensor_1d.MatrixProductOperator(array,shape='lrud',tags=self.qubits,upper_ind_id='k{}',lower_ind_id='b{}',site_tag_id='I{}')


    def get_MPS(self):

        # MPS representation of the initial vacuum state
        tensor0 = np.zeros((1,2))
        tensor0[0,1] = 1 # basis state |1> on the left boundary
        #
        tensor1 = np.zeros((1,1,2))
        tensor1[0,0,1] = 1 # basis state |1> in the bulk
        #
        tensor2 = np.zeros((1,1,2))
        tensor2[0,0,0] = 1 # basis state |0> in the bulk
        #
        tensor3 = np.zeros((1,2))
        tensor3[0,0] = 1 # basis state |0> on the right boundary

        array = [tensor0] + (self.n-1)*[tensor1] +  (self.n-1)*[tensor2] + [tensor3]
        self.mps  = qtn.tensor_1d.MatrixProductState(array,shape='lrp',tags=self.qubits,site_ind_id='k{}',site_tag_id='I{}')

        for k in self.Mu:
            mpo = self.getMPO(k)
            mps_gate_with_mpo_direct(self.mps,mpo, cutoff=self.relerr,cutoff_mode='rsum1',inplace=True)
           


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
