import numpy as np

# NOTE: MPNUM is no longer maintained, but it's still a good package for MPS/MPO simulations!
# The following fixes dependency issues for numpy 2.0
if np.version.version > "2.0":
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
import champs

import quimb.tensor as qtn
from champs.builder import QUIMB_BACKEND, MPNUM_BACKEND


class CharacterBuilder(champs.Builder):
    def __init__(
        self,
        Mu: tuple[int],
        Nu: tuple[int] = (0,),
        relerr: float = 1e-10,
        backend: str = MPNUM_BACKEND,
    ):
        """
        MPS algorithm for characters of the symmetric group S_n described in arXiv:2501.????

        Takes as input a conjugacy class Mu of S_n specified as a list of
        positive integers that sum to n

        Args:
            Mu (tuple[int]): S_n conjugacy class as a list of positive integers that sum up to n.
            relerr (float, optional): MPS compression relative error. Defaults to 1e-10.
        """

        super().__init__(Mu=Mu, Nu=Nu, relerr=relerr, backend=backend)

    def get_character(self, Lambda: tuple[int]) -> int:
        """
        Computes the character chi_Lambda(Mu) for a conjugacy class Mu and an irrep Lambda of S_n
        Note that the conjugacy class Mu is fixed by the CharacterBuilder object. Caches the partial products of MPS matrices over each interval to speed up the computation.

        Args:
            Lambda (tuple[int]): an irrep of S_n as a list of positive integers that sums up to n.

        Returns:
            int: character chi_Lambda(Mu)
        """
        assert len(Lambda) <= self.n

        return int(np.round(self._contract(Lambda)))

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
        tensor = np.zeros((k + 2, 2, 2, k + 2))  # index ordering Left Right Up Down
        tensor[0, :, :, 0] = np.eye(2)
        tensor[k + 1, :, :, k + 1] = np.eye(2)
        tensor[0, :, :, 1] = np.array([[0, 1], [0, 0]])  # flip qubit from '1' to '0'
        tensor[k, :, :, k + 1] = np.array(
            [[0, 0], [1, 0]]
        )  # flip qubit from '0' to '1'

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
            qtn.tensor_1d.MatrixProductOperator: MPO representation of the current operator J_k.
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
            shape="lrud",
            tags=self.qubits,
            upper_ind_id="k{}",
            lower_ind_id="b{}",
            site_tag_id="I{}",
        )

    def get_MPO(self, k: int) -> mp.MPArray | qtn.tensor_1d.MatrixProductOperator:
        """
        MPO representation of the current operator J_k = sum_i a_i a_{i+k}^dag.

        Args:
            k (int): parameter specifying the current operator J_k.

        Returns:
            mpnum.MPArray | qtn.tensor_1d.MatrixProductOperator: MPO representation of the current operator J_k. Return type depends on the self.backend.
        """

        if self.backend == MPNUM_BACKEND:
            return self._get_MPNUM_MPO(k)

        elif self.backend == QUIMB_BACKEND:
            return self._get_QUIMB_MPO(k)
