import itertools
from typing import List


def combine(n: int, k: int) -> List[List[int]]:
    # Tue, 01 Aug 2023  01:39:32
    # time O(r * nCr)
    # space O(r)
    return list(itertools.combinations(range(1, n + 1), k))


def combineRC(n: int, k: int) -> List[List[int]]:
    res = []
    comb = []

    def backtrack(start: int):
        if len(comb) == k:
            res.append(comb.copy())
            return
        for i in range(start, n + 1):
            comb.append(i)
            backtrack(i + 1)
            comb.pop()

    backtrack(1)
    return res


def permute(nums: List[int]) -> List[List[int]]:
    return list(itertools.permutations(nums))


def permuteRC(nums: List[int]) -> List[List[int]]:
    def backtrack():
        if len(curr) == n:
            res.append(curr[:])
        for num in nums:
            if num not in curr:
                curr.append(num)
                backtrack()
                curr.pop()

    n = len(nums)
    res = []
    curr = []
    backtrack()
    return res
