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


def numMusicPlaylists(n: int, goal: int, k: int) -> int:
    mod = 10**9 + 7
    dp = {}

    def count(curr_goal, old_songs):
        if (curr_goal, old_songs) in dp:
            return dp[(curr_goal, old_songs)]
        if curr_goal == 0 and old_songs == n:
            return 1
        if curr_goal == 0 or old_songs > n:
            return 0

        # choose new song
        res = (n - old_songs) * count(curr_goal - 1, old_songs + 1)
        if old_songs > k:
            # choose old song
            res += (old_songs - k) * count(curr_goal - 1, old_songs)
        dp[(curr_goal, old_songs)] = res % mod
        return dp[(curr_goal, old_songs)]

    return count(goal, 0)


def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    def binary_search(arr: List[int]) -> bool:
        l, r = 0, len(arr) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if arr[mid] < target:
                l = mid + 1
            elif arr[mid] == target:
                return True
            else:
                r = mid - 1
        return False

    res = []
    for row in matrix:
        res.extend(row)
    return binary_search(res)


def searchMatrixBS(matrix: List[List[int]], target: int) -> bool:
    top, down = 0, len(matrix)

    while top < down:
        mid = top + (down - top) // 2
        if matrix[mid][-1] < target:
            top = mid + 1
        else:
            down = mid

    if top == len(matrix):
        return False
    l, r = 0, len(matrix[0])
    while l < r:
        mid = l + (r - l) // 2
        if matrix[top][mid] < target:
            l = mid + 1
        else:
            r = mid
    return l < len(matrix[top]) and matrix[top][l] == target


def minimizeMax(nums: List[int], p: int) -> int:
    def is_valid(x):
        i = cnt = 0
        while i < len(nums) - 1:
            if abs(nums[i] - nums[i + 1]) <= x:
                cnt += 1
                i += 2
            else:
                i += 1
            if cnt == p:
                return True
        return False

    if p == 0:
        return 0
    l, r = 0, nums[-1] - nums[0]
    nums.sort()
    while l < r:
        mid = l + (r - l) // 2
        if is_valid(mid):
            r = mid
        else:
            l = mid + 1
    return l
