import numpy as np
import mpnum as mp  # MPS/MPO package
from utils import majorize
import champs

class KostkaBuilder(champs.Builder):
    """
    MPS algorithm for skew Kostka numbers.

    Computes Kostkas for a given weight vector Mu and skew Nu.

    Args:
        Mu (tuple[int]): we assume that Mu is given in non-increasing order
        Nu (tuple[int], optional): _description_. Defaults to (0, ).
        relerr (_type_, optional): _description_. Defaults to 1e-14.
    """

    def __init__(self, Mu: tuple[int], Nu: tuple[int] = (0, ), relerr=1e-14):
        super().__init__(Mu, Nu, relerr=relerr)

    # Computes the skew Kostka K_Lambda\Nu,Mu for a partition Lambda
    # Input:
    # Lambda: a non-increasing list of positive integers summing to n

    def get_kostka(self, Lambda):
        # check majorization or if lambda \ nu is a valid skew partition
        if self.Nu == (0,) and not majorize(self.Mu, Lambda): # standard Kostka
                return 0
        elif not self.valid_skew(Lambda): # skew Kostka
            return 0
        
        return int(np.round(self._Builder__contract(Lambda)))

    # Returns a MPO representing (operator) complete symmetric polynomials
    def get_MPO(self, k):

        array = []
        # index ordering LUDR

        # left boundary
        tensor = np.zeros((1, 2, 2, 2 * k + 1))
        tensor[0, :, :, 0] = np.eye(2)
        tensor[0, :, :, 1] = np.array([[0, 1], [0, 0]])  # annihilate
        array.append(tensor)

        # bulk
        tensor = np.zeros((2 * k + 1, 2, 2, 2 * k + 1))
        for i in range(k - 1):  # runs until k-2
            tensor[2 * i, :, :, 2 * i] = np.eye(2)
            tensor[2 * i + 1, :, :, 2 * i + 2] = np.array([[0, 0], [1, 0]])
            tensor[2 * i + 1, :, :, 2 * i + 3] = np.array([[1, 0], [0, 0]])
            tensor[2 * i, :, :, 2 * i + 1] = np.array([[0, 1], [0, 0]])

        tensor[2 * k - 2, :, :, 2 * k - 2] = np.eye(2)
        tensor[2 * k - 2, :, :, 2 * k - 1] = np.array([[0, 1], [0, 0]])
        tensor[2 * k - 1, :, :, 2 * k] = np.array([[0, 0], [1, 0]])
        tensor[2 * k, :, :, 2 * k] = np.eye(2)

        array = array + (2 * self.m - 2) * [tensor]

        # right boundary
        tensor = np.zeros((2 * k + 1, 2, 2, 1))
        tensor[2 * k, :, :, 0] = np.eye(2)
        tensor[2 * k - 1, :, :, 0] = np.array([[0, 0], [1, 0]])  # create
        array.append(tensor)

        return mp.MPArray(mp.mpstruct.LocalTensors(array))
