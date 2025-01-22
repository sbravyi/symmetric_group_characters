from math import lgamma, exp


def partitions(n: int, minimum_partition_value: int = 1):
    """
    Returns a generator that yields all partitions of n.

    Source:
    https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning

    Args:
        n (int): Integer to partition
        minimum_partition_value (int, optional): Minimum partition value. Defaults to 1.
    """
    yield (n,)
    for i in range(minimum_partition_value, n // 2 + 1):
        for p in partitions(n - i, i):
            yield (i,) + p


def get_partitions(n: int) -> list[tuple[int]]:
    """
    Returns a list of all partitions of n.

    Args:
        n (int): Integer to partition

    Returns:
        list[tuple[int]]: List of all partitions of n.
    """
    return [tuple(reversed(list(p))) for p in list(partitions(n))]


def perm_module_d(mu: tuple[int]) -> int:
    """
    Returns the dimension of the permutation module of label Mu

    Args:
        Mu (tuple[int]): Partition as a list of positive integers in nonincreasing order that sum up to n.

    Returns:
        int: Dimension of the permutation module of label Mu
    """
    val = lgamma(sum(mu) + 1)
    for part in mu:
        val -= lgamma(part + 1)
    return int(round(exp(val)))


def majorize(mu: tuple[int], ell: tuple[int], eq=True) -> bool:
    """
    Determines if lambda >= Mu in majorization order

    Args:
        mu (tuple[int]): Partition as a list of positive integers in nonincreasing order.
        ell (tuple[int]): Partition as a list of positive integers in nonincreasing order.
        eq: Flag for requiring that Mu and Lambda are partitions of the same number

    Returns:
        bool: True if Lambda >= Mu in majorization order, False otherwise.
    """
    sum_mu = 0
    sum_lm = 0

    for i in range(min(len(ell), len(mu))):
        sum_mu += mu[i]
        sum_lm += ell[i]
        if sum_mu > sum_lm:
            return False

    remaining_mu = sum(mu[i:])
    remaining_lm = sum(ell[i:])
    if eq:
        return sum_mu + remaining_mu == sum_lm + remaining_lm
    return True
