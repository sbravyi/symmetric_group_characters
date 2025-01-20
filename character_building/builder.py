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

class Builder(): 
    """
    Abstract class to define the structure of the builders for the 
    Kostka numbers and characters of the symmetric group S_n.

    Args:
        Mu (tuple[int]): S_n conjugacy class as a list of positive integers that sum up to n.
        relerr (float, optional): MPS compression relative error. Defaults to 1e-10.
    """
    def __init__(self, Mu: tuple[int], relerr: float = 1e-10):
        
        self.Mu = Mu
        self.n = np.sum(self.Mu)
        self.relerr = relerr  # relative error for MPS compression

    def get_MPS(self) -> mp.MPArray | qtn.tensor_1d.MatrixProductState:
        """
        Compute the MPS that encodes all characters of Mu.

        Returns:
            mp.MPArray | qtn.tensor_1d.MatrixProductState: MPS that encodes all characters of Mu.
        """
        raise NotImplementedError
    
    def get_MPO(self, k:int) -> mp.MPArray | qtn.tensor_1d.MatrixProductOperator:
        """
        MPO representation of the current operator J_k = sum_i a_i a_{i+k}^dag.

        Args:
            k (int): parameter specifying the current operator J_k.

        Returns:
            mpnum.MPArray: MPO representation of the current operator J_k.
        """
        raise NotImplementedError