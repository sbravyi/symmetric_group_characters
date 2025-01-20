from math import lgamma, exp
import numpy as np


def partitions(n:int, I:int=1):
    """
    Returns a generator that yields all partitions of n.

    Source:
    https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning

    Args:
        n (int): Integer to partition
        I (int, optional): Minimum partition value. Defaults to 1.
    """
    yield (n,)
    for i in range(I, n // 2 + 1):
        for p in partitions(n - i, i):
            yield (i,) + p


def get_partitions(n:int) -> list[tuple[int]]:
    """
    Returns a list of all partitions of n.

    Args:
        n (int): Integer to partition

    Returns:
        list[tuple[int]]: List of all partitions of n.
    """
    Pn = [list(p) for p in list(partitions(n))]
    for p in Pn:
        p.reverse()
    return [tuple(p) for p in Pn]

  
def perm_module_d(Mu:tuple[int]) -> int:
    """
    Returns the dimension of the permutation module of label Mu

    Args:
        Mu (tuple[int]): Partition as a list of positive integers in nonincreasing order that sum up to n.

    Returns:
        int: Dimension of the permutation module of label Mu
    """
    val = lgamma(sum(Mu)+1)
    for part in Mu:
        val -= lgamma(part+1)
    return int(round(exp(val)))  

def majorize(Mu: tuple[int], Lambda: tuple[int], Eq = True) -> bool:
    """
    Determines if lambda >= Mu in majorization order

    Args:
        Mu (tuple[int]): Partition as a list of positive integers in nonincreasing order.
        Lambda (tuple[int]): Partition as a list of positive integers in nonincreasing order.
        Eq: Flag for requiring that Mu and Lambda are partitions of the same number

    Returns:
        bool: True if Lambda >= Mu in majorization order, False otherwise.
    """
    sum_mu = 0
    sum_lm = 0

    for i in range(min(len(Lambda), len(Mu))):
        sum_mu += Mu[i]
        sum_lm += Lambda[i]
        if sum_mu > sum_lm:
            return False
    
    if Eq and sum_mu + sum(Mu[i:]) == sum_lm + sum(Lambda[i:]) or not Eq:
        return True
    else:
        return False
