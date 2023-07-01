import itertools
from typing import List


def threeSum(nums: List[int]) -> List[List[int]]:
    pass


def nextPermutation(nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    pass


def letterCombinations(digits: str) -> List[str]:
    digits_to_letters = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv',
                         '9': 'wxyz'}

    return [''.join(tup) for tup in itertools.product(*(digits_to_letters[digit] for digit in digits))]


def letterCombinationsNeetCode(digits: str) -> List[str]:
    digits_to_letters = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv',
                         '9': 'wxyz'}

    def backtrack(i: int, path: str):
        if len(path) == len(digits):
            res.append(path)
            return
        for c in digits_to_letters[digits[i]]:
            backtrack(i + 1, path + c)

    res = []
    if digits:
        backtrack(0, '')
    return res
