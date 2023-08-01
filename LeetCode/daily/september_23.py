import itertools
from typing import List


def combine(n: int, k: int) -> List[List[int]]:
    # Tue, 01 Aug 2023  01:39:32
    # time O(r * nCr)
    # space O(r)
    return list(itertools.combinations(range(1, n + 1), k))
