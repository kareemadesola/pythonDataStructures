import itertools
from typing import List, Optional

from LeetCode.Biweekly.contest_82 import TreeNode


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


def letterCombinations(digits: str) -> List[str]:
    n = len(digits)

    def backtrack(i: int):
        if len(curr) == n:
            res.append("".join(curr[:]))
            return
        for c in digit_to_str[digits[i]]:
            curr.append(c)
            backtrack(i + 1)
            curr.pop()

    digit_to_str = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }
    curr = []
    if digits:
        backtrack(0)
    res = []
    return res


def wordBreak(s: str, wordDict: List[str]) -> bool:
    dp = [False] * (len(s) + 1)
    dp[len(s)] = True

    for i in range(len(s) - 1, -1, -1):
        for w in wordDict:
            if s[i : i + len(w)] == w:
                dp[i] = dp[i + len(w)]
            if dp[i]:
                break
    return dp[0]


def generateTrees(n: int) -> List[Optional[TreeNode]]:
    dp = {}

    def generate(l: int, r: int) -> List[Optional[TreeNode]]:
        if l > r:
            return [None]
        if (l, r) in dp:
            return dp[(l, r)]

        res = []
        for val in range(l, r + 1):
            for l_tree in generate(l, val - 1):
                for r_tree in generate(val + 1, r):
                    root = TreeNode(val, l_tree, r_tree)
                    res.append(root)

        dp[(l, r)] = res
        return res

    return generate(1, n)
