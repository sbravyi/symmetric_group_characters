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
from character_building.builder import Builder


class SkewCharacterBuilder(Builder):
    def __init__(self, Mu: tuple[int], Nu:tuple[int], relerr:float = 1e-10):
        """
        MPS algorithm for skew characters of the symmetric group S_n described in arXiv.2501.????

        Args:
        Mu (tuple[int]): _description_
        relerr (float, optional): _description_. Defaults to 1e-10.
        """
        super().__init__(Mu, Nu, relerr)

        self.maximum_rank = 1
        self.tensor1 = np.zeros((1, 2, 1))
        self.tensor1[0, 1, 0] = 1 # basis state |1>
        self.tensor0 = np.zeros((1, 2, 1))
        self.tensor0[0, 0, 0] = 1 # basis state |0>

        self.mps = self.get_MPS()
        self.MPSready = True

    

    def get_MPS(self) -> mp.MPArray:
        array = []  # Local tensors
        # Traverse Nu in reverse order
        array += [self.tensor0] * self.Nu[self.m - 1]  # step right
        array += [self.tensor1]  # step up
        for i in range(self.m - 1, 0, -1):
            array += [self.tensor0] * \
                (self.Nu[i - 1] - self.Nu[i])  # step right
            array += [self.tensor1]  # step up
        array = array + [self.tensor0] * \
            (2 * self.m - len(array))  # step right

