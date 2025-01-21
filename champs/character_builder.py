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
from champs.builder import Builder

import quimb.tensor as qtn
from champs.builder import Builder, QUIMB_BACKEND, MPNUM_BACKEND, MPNUM_DOWN, MPNUM_UP


class CharacterBuilder(Builder):
    def __init__(self, Mu: tuple[int], Nu:tuple[int] = (0, ), relerr: float = 1e-10, backend: str = MPNUM_BACKEND):
        """
        MPS algorithm for characters of the symmetric group S_n described in arXiv:2501.????

        Takes as input a conjugacy class Mu of S_n specified as a list of
        positive integers that sum to n

        Args:
            Mu (tuple[int]): S_n conjugacy class as a list of positive integers that sum up to n.
            relerr (float, optional): MPS compression relative error. Defaults to 1e-10.
        """

        super().__init__(Mu=Mu, Nu=Nu, relerr=relerr, backend=backend)
        
        # maximum MPS bond dimension (maximum Schmidt rank)
        self.maximum_rank = 1

        self.tensor1 = np.zeros((1, 2, 1))
        self.tensor1[0, 1, 0] = 1  # basis state |1>
        self.tensor0 = np.zeros((1, 2, 1))
        self.tensor0[0, 0, 0] = 1  # basis state |0>

        # compute the MPS that encodes all characters of Mu
        self.mps = self.get_MPS()

        # Caching registers for partial products of MPS matrices.
        # Divide the spin chain into four intervals: left (L), center left
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

    def get_character(self, Lambda: tuple[int]) -> int:
        """
        Computes the character chi_Lambda(Mu) for a conjugacy class Mu and an irrep Lambda of S_n
        Note that the conjugacy class Mu is fixed by the CharacterBuilder object.

        Caches the partial products of MPS matrices over each interval to speed up the computation.

        Args:
            Lambda (tuple[int]): an irrep of S_n as a list of positive integers that sums up to n.

        Returns:
            int: character chi_Lambda(Mu)
        """
        assert (len(Lambda) <= self.n)
        # pad the partition Lambda with zeros to make n parts
        # TODO: check 
        padded_Lambda = list(Lambda) + [0] * (self.m - len(Lambda))
        if self.m < 8:
            # don't use caching for small n's
            array = [MPNUM_DOWN] * (2 * self.m)
            for i in range(self.m):
                array[padded_Lambda[i] + self.m - 1 - i] = MPNUM_UP
            basis_state_mps = mp.MPArray(mp.mpstruct.LocalTensors(array))
            # compute inner product between a basis state and the MPS
            return int(np.round(mp.mparray.inner(basis_state_mps, self.mps)))

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
        
        return int(np.round(chi[0][0]))
    

    def _get_MPNUM_MPO(self, k: int) -> mp.MPArray:
        """
        MPO representation of the current operator J_k = sum_i a_i a_{i+k}^dag.

        Uses the MPNUM package to build the MPO.

        Args:
            k (int): parameter specifying the current operator J_k.

        Returns:
            mp.MPArray: MPO representation of the current operator J_k.
        """
        array = []

        # left boundary
        tensor = np.zeros((1, 2, 2, k + 2))
        tensor[0, :, :, 0] = np.eye(2)
        tensor[0, :, :, 1] = np.array([[0, 1], [0, 0]])  # flip qubit from '1' to '0'
        array.append(tensor)

        # bulk
        tensor = np.zeros((k + 2, 2, 2, k + 2))   # index ordering Left Right Up Down
        tensor[0, :, :, 0] = np.eye(2)
        tensor[k + 1, :, :, k + 1] = np.eye(2)
        tensor[0, :, :, 1] = np.array([[0, 1], [0, 0]])  # flip qubit from '1' to '0'
        tensor[k, :, :, k + 1] = np.array([[0, 0], [1, 0]])  # flip qubit from '0' to '1'

        # Pauli Z
        for j in range(1, k):
            tensor[j, :, :, j + 1] = np.array([[1, 0], [0, -1]])

        array = array + (2 * self.m - 2) * [tensor]

        # right boundary
        tensor = np.zeros((k + 2, 2, 2, 1))
        tensor[k + 1, :, :, 0] = np.eye(2)
        # flip qubit from '0' to '1'
        tensor[k, :, :, 0] = np.array([[0, 0], [1, 0]])
        array.append(tensor)
        return mp.MPArray(mp.mpstruct.LocalTensors(array))
    

    def _get_QUIMB_MPO(self, k: int) -> qtn.tensor_1d.MatrixProductOperator:
        """
        MPO representation of the current operator J_k = sum_i a_i a_{i+k}^dag.

        Uses the QUIMB package to build the MPO.

        Args:
            k (int): parameter specifying the current operator J_k.

        Returns:
            mp.MPArray: MPO representation of the current operator J_k.
        """
        array = []

        # left boundary
        tensor = np.zeros((k + 2, 2, 2))
        tensor[0, :, :] = np.eye(2)
        # flip qubit from '1' to '0'
        tensor[1, :, :] = np.array([[0, 1], [0, 0]])
        array.append(tensor)

        # bulk
        tensor = np.zeros((k + 2, k + 2, 2, 2))
        tensor[0, 0, :, :] = np.eye(2)
        tensor[k + 1, k + 1, :, :] = np.eye(2)
        # flip qubit from '1' to '0'
        tensor[0, 1, :, :] = np.array([[0, 1], [0, 0]])
        # flip qubit from '0' to '1'
        tensor[k, k + 1, :, :] = np.array([[0, 0], [1, 0]])

        # Pauli Z
        for j in range(1, k):
            tensor[j, j + 1, :, :] = np.array([[1, 0], [0, -1]])

        array = array + (2 * self.n - 2) * [tensor]

        # right boundary
        tensor = np.zeros((k + 2, 2, 2))
        tensor[k + 1, :, :] = np.eye(2)
        # flip qubit from '0' to '1'
        tensor[k, :, :] = np.array([[0, 0], [1, 0]])
        array.append(tensor)

        return qtn.tensor_1d.MatrixProductOperator(
            array,
            shape='lrud',
            tags=self.qubits,
            upper_ind_id='k{}',
            lower_ind_id='b{}',
            site_tag_id='I{}')


    def get_MPO(self, k: int) -> mp.MPArray | qtn.tensor_1d.MatrixProductOperator:
        """
        MPO representation of the current operator J_k = sum_i a_i a_{i+k}^dag.

        Args:
            k (int): parameter specifying the current operator J_k.

        Returns:
            mpnum.MPArray: MPO representation of the current operator J_k.
        """

        if self.backend == MPNUM_BACKEND:
            return self._get_MPNUM_MPO(k)
        
        elif self.backend == QUIMB_BACKEND: 
            return self._get_QUIMB_MPO(k)
            
