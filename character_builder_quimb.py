import numpy as np
import quimb.tensor as qtn
from quimb.tensor.tensor_1d_compress import mps_gate_with_mpo_direct


class CharacterBuilder:

    def __init__(self, Mu: tuple[int], relerr: float = 1e-10):
        """
        MPS algorithm for characters of the symmetric group S_n described in arXiv 2501.

        Args:
            Mu (tuple[int]): S_n conjugacy class as a list of positive integers that sum up to n.
            relerr (float, optional): Relative error for MPS compression. Defaults to 1e-10.
        """
        self.Mu = list(np.sort(Mu))
        self.n = np.sum(self.Mu)
        self.relerr = relerr
        self.qubits = [i for i in range(2 * self.n)]
        self.get_MPS()  # compute the MPS that encodes all characters of Mu

    def get_character(self, Lambda: tuple[int]) -> int:
        """
        Computes the character chi_Lambda(Mu) for a conjugacy class Mu and an irrep Lambda of S_n
        Note that the conjugacy class Mu is fixed by the CharacterBuilder object.

        Args:
            Lambda (tuple[int]): an irrep of S_n as a list of positive integers that sums up to n.

        Returns:
            int: character chi_Lambda(Mu)
        """
        assert (len(Lambda) <= self.n)
        padded_Lambda = list(Lambda) + [0] * (self.n - len(Lambda))
        bit_string = np.zeros(2 * self.n, dtype=int)
        supp = [padded_Lambda[i] + self.n - 1 - i for i in range(self.n)]
        bit_string[supp] = 1
        return self.mps.amplitude(bit_string)

    def getMPO(self, k: int) -> qtn.tensor_1d.MatrixProductOperator:
        """
        MPO representation of the current operator J_k = sum_i a_i a_{i+k}^dag.
        See Crichigno and Prakash; arXiv:2404.04322 or arXiv:2501.

        Args:
            k (int): parameter specifying the current operator J_k.

        Returns:
            qtn.tensor_1d.MatrixProductOperator: MPO representation of the current operator J_k.
        """

        array = []

        tensor = np.zeros((k + 2, 2, 2))  # left boundary
        tensor[0, :, :] = np.eye(2)

        # flip qubit from '1' to '0'
        tensor[1, :, :] = np.array([[0, 1], [0, 0]])
        array.append(tensor)

        tensor = np.zeros((k + 2, k + 2, 2, 2))  # bulk
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

        tensor = np.zeros((k + 2, 2, 2))  # right boundary
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

    def get_MPS(self):


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
        self.mps = qtn.tensor_1d.MatrixProductState(
            array, shape='lrp', tags=self.qubits,
            site_ind_id='k{}', site_tag_id='I{}')

        for k in self.Mu:
            mpo = self.getMPO(k)
            mps_gate_with_mpo_direct(
                self.mps,
                mpo,
                cutoff=self.relerr,
                cutoff_mode='rsum1',
                inplace=True)
            
        # TODO: compress the MPS
        # TODO: make this return. 
        return self.mps 

    