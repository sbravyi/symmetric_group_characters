import numpy as np
import quimb.tensor as qtn
from character_building.builder import Builder, QUIMB_BACKEND


class CharacterBuilder(Builder):
    """
    MPS algorithm for characters of the symmetric group S_n described in arXiv:2501.?????
    """

    def __init__(self, Mu: tuple[int], relerr: float = 1e-10):
        """
        MPS algorithm for characters of the symmetric group S_n described in arXiv:2501.?????

        Args:
            Mu (tuple[int]):
        """
        super().__init__(Mu, relerr=relerr, backend=QUIMB_BACKEND)

        # compute the MPS that encodes all characters of Mu
        self.mps = self.get_MPS()

        self.qubits = [i for i in range(2 * self.n)]
        # local tensors
        self.lt = []
        # left boundary
        tensor = np.zeros(
            (1, self.mps.arrays[0].shape[0], self.mps.arrays[0].shape[1]))
        tensor[0, :, :] = self.mps.arrays[0]
        self.lt.append(tensor)
        # interior
        self.lt = self.lt + [self.mps.arrays[q]
                             for q in range(1, 2 * self.n - 1)]
        # right boundary
        tensor = np.zeros(
            (self.mps.arrays[0].shape[0], 1, self.mps.arrays[0].shape[1]))
        tensor[:, 0, :] = self.mps.arrays[-1]
        self.lt.append(tensor)

        # divide the spin chain into four intervals: left (L), center left
        # (C1), center right C2, right (R)
        self.n1 = int(np.round(self.n / 2))
        self.n2 = self.n
        self.n3 = int(np.round(3 * self.n / 2))
        self.L = [i for i in range(2 * self.n) if i < self.n1]
        self.C1 = [
            i for i in range(
                2 *
                self.n) if i >= self.n1 and i < self.n2]
        self.C2 = [
            i for i in range(
                2 *
                self.n) if i >= self.n2 and i < self.n3]
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

        Args:
            Lambda (tuple[int]): irrep of S_n

        Returns:
            int: character chi_Lambda
        """
        assert (len(Lambda) <= self.n)
        padded_Lambda = list(Lambda) + [0] * (self.n - len(Lambda))
        bitstring = np.zeros(2 * self.n, dtype=int)
        supp = [padded_Lambda[i] + self.n - 1 - i for i in range(self.n)]
        bitstring[supp] = 1
        if self.n < 8:
            # don't use caching for small n's
            return self.mps.amplitude(bitstring)

        # project bitstring onto each caching register
        xL = bitstring[self.L]
        xC1 = bitstring[self.C1]
        xC2 = bitstring[self.C2]
        xR = bitstring[self.R]

        if not (tuple(xL) in self.cacheL):
            self.cacheL[tuple(xL)] = np.linalg.multi_dot(
                [self.lt[self.L[i]][:, :, xL[i]] for i in range(self.nL)])

        if not (tuple(xC1) in self.cacheC1):
            self.cacheC1[tuple(xC1)] = np.linalg.multi_dot(
                [self.lt[self.C1[i]][:, :, xC1[i]] for i in range(self.nC1)])

        if not (tuple(xC2) in self.cacheC2):
            self.cacheC2[tuple(xC2)] = np.linalg.multi_dot(
                [self.lt[self.C2[i]][:, :, xC2[i]] for i in range(self.nC2)])

        if not (tuple(xR) in self.cacheR):
            self.cacheR[tuple(xR)] = np.linalg.multi_dot(
                [self.lt[self.R[i]][:, :, xR[i]] for i in range(self.nR)])

        chi = (self.cacheL[tuple(xL)] @ self.cacheC1[tuple(xC1)]
               ) @ (self.cacheC2[tuple(xC2)] @ self.cacheR[tuple(xR)])
        
        return chi[0][0]

    def get_MPO(self, k) -> qtn.tensor_1d.MatrixProductOperator:
        """
        MPO representation of the current operator J_k = sum_i a_i a_{i+k}^dag.

        Args:
            k (int): parameter specifying the current operator J_k.

        Returns:
            mpnum.MPArray: MPO representation of the current operator J_k.
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