import numpy as np

# NOTE: MPNUM is no longer maintained, but it's still a good package for MPS/MPO simulations!
# The following fixes dependency issues for numpy 2.0
if np.version.version > '2.0':
    np.float_ = np.float64
    np.complex_ = np.complex128

# The following fixes dependency issues for python >= 3.7
import sys
import collections
if sys.version_info[0] >= 3 and sys.version_info[1] >= 7:
    collections.Sequence = collections.abc.Sequence
    collections.Iterable = collections.abc.Iterable
    collections.Iterator = collections.abc.Iterator

import mpnum as mp  # MPS/MPO simulation package
import quimb.tensor as qtn

MPNUM_BACKEND = 'mpnum'
QUIMB_BACKEND = 'quimb'

class Builder():
    """
    Abstract class to define the structure of the builders for the 
    Kostka numbers and characters of the symmetric group S_n.

    Args:
        Mu (tuple[int]): S_n conjugacy class as a list of positive integers that sum up to n.
        Nu (tuple[int], optional): S_n conjugacy class as a list of positive integers that sum up to n. Defaults to (0,). Used to encode skew partitions Mu/Nu.
        relerr (float, optional): MPS compression relative error. Defaults to 1e-10.
    """
    def __init__(self, 
                 Mu: tuple[int], 
                 Nu:tuple[int] = (0, ), 
                 relerr: float = 1e-10, 
                 backend: str = MPNUM_BACKEND):
        self.Mu = Mu
        self.Nu = Nu
        self.n = np.sum(self.Mu)
        self.m = self.n + np.sum(self.Nu) # length of the skew partition
        self.relerr = relerr  # relative error for MPS compression
        self.backend = backend 


    def get_MPS(self) -> mp.MPArray | qtn.tensor_1d.MatrixProductState:
        """
        Compute the MPS that encodes all characters of Mu.

        Returns:
            mp.MPArray | qtn.tensor_1d.MatrixProductState: MPS that encodes all characters of Mu.
        """
        raise NotImplementedError
    

    def get_initial_MPS(self) -> mp.MPArray | qtn.tensor_1d.MatrixProductState:
        """
        Compute the MPS that encodes the initial state. 

        Returns:
            mp.MPArray | qtn.tensor_1d.MatrixProductState: _description_
        """

        if self.backend == MPNUM_BACKEND:
            tensor1:np.array = np.zeros((1, 2, 1))
            tensor1[0, 1, 0] = 1

            tensor0:np.array = np.zeros((1, 2, 1))
            tensor0[0, 0, 0] = 1

            array = []  # Local tensors
            # Traverse Nu in reverse order
            array += [tensor0] * self.Nu[self.m - 1]  # step right
            array += [tensor1]  # step up
            for i in range(self.m - 1, 0, -1):
                array += [tensor0] * \
                    (self.Nu[i - 1] - self.Nu[i])  # step right
                array += [tensor1]  # step up
            array = array + [tensor0] * \
                (2 * self.m - len(array))  # step right
            return mp.MPArray(mp.mpstruct.LocalTensors(array))
        
        elif self.backend == QUIMB_BACKEND:
            # MPS representation of the initial vacuum state
            tensor0 = np.zeros((1, 2))
            tensor0[0, 1] = 1  # basis state |1> on the left boundary
            #
            tensor1 = np.zeros((1, 1, 2))
            tensor1[0, 0, 1] = 1  # basis state |1> in the bulk
            #
            tensor2 = np.zeros((1, 1, 2))
            tensor2[0, 0, 0] = 1  # basis state |0> in the bulk
            #
            tensor3 = np.zeros((1, 2))
            tensor3[0, 0] = 1  # basis state |0> on the right boundary

            array = [tensor0] + (self.n - 1) * [tensor1] + \
                (self.n - 1) * [tensor2] + [tensor3]
            
            return qtn.tensor_1d.MatrixProductState(
                array, shape='lrp', tags=self.qubits, site_ind_id='k{}', site_tag_id='I{}')


    

    def get_MPO(self, k:int) -> mp.MPArray | qtn.tensor_1d.MatrixProductOperator:
        """
        MPO representation of the current operator J_k = sum_i a_i a_{i+k}^dag.

        Args:
            k (int): parameter specifying the current operator J_k.

        Returns:
            mpnum.MPArray: MPO representation of the current operator J_k.
        """
        raise NotImplementedError