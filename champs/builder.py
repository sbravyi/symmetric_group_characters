#    ___ _                               
#   / __\ |__   __ _ _ __ ___  _ __  ___ 
#  / /  | '_ \ / _` | '_ ` _ \| '_ \/ __|
# / /___| | | | (_| | | | | | | |_) \__ \
# \____/|_| |_|\__,_|_| |_| |_| .__/|___/
#                             |_|        



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

# ----------------- QUIMB Imports -----------------
import quimb.tensor as qtn
from quimb.tensor.tensor_1d_compress import mps_gate_with_mpo_direct

# ----------------- MPNUM Imports -----------------
import mpnum as mp  # MPS/MPO simulation package
from utils import majorize


# ----------------- MPNUM Constants -----------------
MPNUM_BACKEND = "mpnum"

# Basis states for the mpnum backend
MPNUM_UP: np.array = np.zeros((1, 2, 1))
MPNUM_UP[0, 1, 0] = 1

MPNUM_DOWN: np.array = np.zeros((1, 2, 1))
MPNUM_DOWN[0, 0, 0] = 1

# ----------------- QUIMB Constants -----------------
QUIMB_BACKEND = "quimb"

# Basis states for the quimb backend
QUIMB_UP_BOUNDARY: np.array = np.zeros((1, 2))
QUIMB_UP_BOUNDARY[0, 1] = 1

QUIMB_UP_BULK: np.array = np.zeros((1, 1, 2))
QUIMB_UP_BULK[0, 0, 1] = 1

QUIMB_DOWN_BULK: np.array = np.zeros((1, 1, 2))
QUIMB_DOWN_BULK[0, 0, 0] = 1

QUIMB_DOWN_BOUNDARY: np.array = np.zeros((1, 2))
QUIMB_DOWN_BOUNDARY[0, 0] = 1


class Builder:
    """
    Defines the structure of the builders for
    Kostka numbers and Sn characters.

    Args:
        Mu (tuple[int]): S_n conjugacy class as a list of positive integers that sum up to n.
        Nu (tuple[int], optional): S_n conjugacy class as a list of positive integers that sum up to n. Defaults to (0,). Used to encode skew partitions Mu/Nu.
        relerr (float, optional): MPS compression relative error. Defaults to 1e-10.
    """

    def __init__(
        self,
        Mu: tuple[int],
        Nu: tuple[int] = (0,),
        relerr: float = 1e-10,
        backend: str = MPNUM_BACKEND,
    ):
        self.Mu = Mu
        self.Nu = Nu
        self.n = np.sum(self.Mu)
        # size of Lambdas to evaluate will be n + sum(Nu)
        self.m = self.n + np.sum(self.Nu)  # size of Hilbert space
        self.Nu = Nu + (0,) * (self.m - len(Nu))  # pad Nu with 0s

        self.relerr = relerr  # relative error for MPS compression
        self.backend = backend
        self.maximum_rank = 1

        self.qubits = [
            i for i in range(2 * self.m)
        ]  # used by quimb to label the qubits

        # compute the MPS that encodes all characters of Mu
        self.mps = self.get_MPS()

        # CACHING:

        # divide the spin chain into four intervals: left (L), center left
        # (C1), center right C2, right (R)
        self.m1 = int(np.round(self.m / 2))
        self.m2 = self.m
        self.m3 = int(np.round(3 * self.m / 2))
        self.L = [i for i in range(2 * self.m) if i < self.m1]
        self.C1 = [i for i in range(2 * self.m) if i >= self.m1 and i < self.m2]
        self.C2 = [i for i in range(2 * self.m) if i >= self.m2 and i < self.m3]
        self.R = [i for i in range(2 * self.m) if i >= self.m3]
        self.mL = len(self.L)
        self.mC1 = len(self.C1)
        self.mC2 = len(self.C2)
        self.mR = len(self.R)
        # cache partial products of MPS matrices over each interval
        self.cacheL = {}
        self.cacheC1 = {}
        self.cacheC2 = {}
        self.cacheR = {}

    def get_MPS(self) -> mp.MPArray | qtn.tensor_1d.MatrixProductState:
        """
        Compute the MPS that encodes all characters of Mu.

        Returns:
            mp.MPArray | qtn.tensor_1d.MatrixProductState: MPS that encodes all characters of Mu. The return type depends on self.backend.
        """

        self.maximum_rank = 1
        mps = self.get_initial_MPS()

        if self.backend == MPNUM_BACKEND:
            for k in self.Mu:
                mpo = self.get_MPO(k)
                mps = mp.dot(mpo, mps)
                mps.compress(method="svd", relerr=self.relerr)
                self.maximum_rank = max(self.maximum_rank, np.max(mps.ranks))

        elif self.backend == QUIMB_BACKEND:
            for k in self.Mu:
                mpo = self.get_MPO(k)
                mps_gate_with_mpo_direct(
                    mps, mpo, cutoff=self.relerr, cutoff_mode="rsum1", inplace=True
                )
                for q in self.qubits:
                    if q == 0 or q == (2 * self.m - 1):
                        D = mps.arrays[q].shape[0]
                    else:
                        D = max(mps.arrays[q].shape[0], mps.arrays[q].shape[1])
                    self.maximum_rank = max(D, self.maximum_rank)
        return mps

    def get_bond_dimension(self) -> int:
        """
        Returns the maximum bond dimension (maximum Schmidt rank) of the MPS.

        Returns:
            int: _description_
        """
        return self.maximum_rank

    def get_initial_MPS(self) -> mp.MPArray | qtn.tensor_1d.MatrixProductState:
        """
        Compute the MPS that encodes the initial state.

        Returns:
            mp.MPArray | qtn.tensor_1d.MatrixProductState: an MPS tensor representation of the initial state. The return type depends on self.backend.
        """

        if self.backend == MPNUM_BACKEND:
            array = []  # Local tensors
            # Traverse Nu in reverse order
            array += [MPNUM_DOWN] * self.Nu[self.m - 1]  # step right
            array += [MPNUM_UP]  # step up
            for i in range(self.m - 1, 0, -1):
                array += [MPNUM_DOWN] * (self.Nu[i - 1] - self.Nu[i])  # step right
                array += [MPNUM_UP]  # step up
            array = array + [MPNUM_DOWN] * (2 * self.m - len(array))  # step right
            return mp.MPArray(mp.mpstruct.LocalTensors(array))

        # NOTE: for Nu = (0, ) the initial state is the vacuum state
        # MPS representation of the initial state |1^n 0^n>
        # array = self.n * [self.tensor1] + self.n * [self.tensor0]
        # mps = mp.MPArray(mp.mpstruct.LocalTensors(array))

        elif self.backend == QUIMB_BACKEND:
            if sum(self.Nu) != 0:
                raise NotImplementedError(
                    "Initial state for skew partitions with QUIMB backend is not implemented yet."
                )

            array = (
                [QUIMB_UP_BOUNDARY]
                + (self.m - 1) * [QUIMB_UP_BULK]
                + (self.m - 1) * [QUIMB_DOWN_BULK]
                + [QUIMB_DOWN_BOUNDARY]
            )

            return qtn.tensor_1d.MatrixProductState(
                array,
                shape="lrp",
                tags=self.qubits,
                site_ind_id="k{}",
                site_tag_id="I{}",
            )

    def get_MPO(self, k: int) -> mp.MPArray | qtn.tensor_1d.MatrixProductOperator:
        """
        MPO representation of the current operator J_k = sum_i a_i a_{i+k}^dag.

        Args:
            k (int): parameter specifying the current operator J_k.

        Returns:
            mpnum.MPArray | qtn.tensor_1d.MatrixProductOperator: MPO representation of the current operator J_k. the return type depends on self.backend.
        """
        raise NotImplementedError

    def valid_skew(self, Lambda: tuple[int]) -> bool:
        """
        Determines if Lamba \ Nu has enough boxes to have weight Mu.

        Args:
            Lambda (tuple[int]): partition that defines the skew shape Lambda \ Nu.

        Returns:
            bool: True if Lambda \ Nu has enough boxes to have weight Mu, False otherwise.
        """
        return majorize(self.Nu, Lambda, eq=False) and sum(Lambda) == self.m

    def _contract(self, Lambda: tuple[int]):
        """

        Args:
            Lambda (tuple[int]): partition that defines the skew shape Lambda \ Nu

        Returns:
            float: the amplitude <x_lambda+tau|mps> in the mps

        """

        # Lambda must be a partition of m
        assert sum(Lambda) == self.m

        padded_Lambda = list(Lambda) + [0] * (self.m - len(Lambda))

        if self.m < 8:
            # don't use caching for small m's
            array = [MPNUM_DOWN] * (2 * self.m)
            for i in range(self.m):
                array[padded_Lambda[i] + self.m - 1 - i] = MPNUM_UP
            basis_state_mps = mp.MPArray(mp.mpstruct.LocalTensors(array))
            # compute inner product between a basis state and the MPS
            return mp.mparray.inner(basis_state_mps, self.mps)

        bitstring = np.zeros(2 * self.m, dtype=int)
        supp = [padded_Lambda[i] + self.m - i - 1 for i in range(self.m)]
        bitstring[supp] = 1
        # project bitstring onto each caching register
        xL = bitstring[self.L]
        xC1 = bitstring[self.C1]
        xC2 = bitstring[self.C2]
        xR = bitstring[self.R]

        if not (tuple(xL) in self.cacheL):
            self.cacheL[tuple(xL)] = np.linalg.multi_dot(
                [self.mps.lt[self.L[i]][:, xL[i], :] for i in range(self.mL)]
            )

        if not (tuple(xC1) in self.cacheC1):
            self.cacheC1[tuple(xC1)] = np.linalg.multi_dot(
                [self.mps.lt[self.C1[i]][:, xC1[i], :] for i in range(self.mC1)]
            )

        if not (tuple(xC2) in self.cacheC2):
            self.cacheC2[tuple(xC2)] = np.linalg.multi_dot(
                [self.mps.lt[self.C2[i]][:, xC2[i], :] for i in range(self.mC2)]
            )

        if not (tuple(xR) in self.cacheR):
            self.cacheR[tuple(xR)] = np.linalg.multi_dot(
                [self.mps.lt[self.R[i]][:, xR[i], :] for i in range(self.mR)]
            )

        chi = (self.cacheL[tuple(xL)] @ self.cacheC1[tuple(xC1)]) @ (
            self.cacheC2[tuple(xC2)] @ self.cacheR[tuple(xR)]
        )
        return chi[0][0]
