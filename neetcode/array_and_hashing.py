import heapq
from collections import Counter, defaultdict
from typing import List


def containsDuplicate(nums: List[int]) -> bool:
    seen = set()
    for val in nums:
        if val in seen:
            return True
        seen.add(val)
    return False


def isAnagram(s: str, t: str) -> bool:
    s_cnt = Counter(s)
    for t_val in t:
        s_cnt[t_val] -= 1

    for val in s_cnt.values():
        if val: return False
    return True


def twoSum(nums: List[int], target: int) -> List[int]:
    seen = {}
    for idx, val in enumerate(nums):
        if target - val in seen:
            return [seen[target - val], idx]
        seen[val] = idx


def groupAnagrams(strs: List[str]) -> List[List[str]]:
    res = defaultdict(list)
    for val in strs:
        res[tuple(sorted(val))].append(val)
    return res.values()


def topKFrequent(nums: List[int], k: int) -> List[int]:
    cnt = Counter(nums)
    heap = [(-val, key) for key, val in cnt.items()]
    heapq.heapify(heap)

    res = []
    for _ in range(k):
        res.append(heapq.heappop(heap)[1])
    return res


def productExceptSelf(nums: List[int]) -> List[int]:
    n = len(nums)
    res = [1] * n

    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]

    suffix = 1
    for i in range(n - 1, -1, -1):
        res[i] *= suffix
        suffix *= nums[i]
    return res


def isValidSudoku(board: List[List[str]]) -> bool:
    row_dict, col_dict, grid_dict = defaultdict(list), defaultdict(list), defaultdict(list)

    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == ".":
                continue
            grid_r, grid_c = r // 3, c // 3
            if val in row_dict[r] or val in col_dict[c] or val in grid_dict[(grid_r, grid_c)]:
                return False
            row_dict[r].append(val)
            col_dict[c].append(val)
            grid_dict[(grid_r, grid_c)].append(val)
    return True


def longestConsecutive(nums: List[int]) -> int:
    nums = sorted(set(nums))
    n = len(nums)
    res = 1 if nums else 0
    tmp = 1
    for i in range(n - 1):
        if nums[i] + 1 == nums[i + 1]:
            tmp += 1
        else:
            tmp = 1
        res = max(res, tmp)
    return res


def longestConsecutiveAlt(nums: List[int]) -> int:
    nums = set(nums)
    longest = 0
    for val in nums:
        if val - 1 in nums:
            continue
        length = 1
        while val + length in nums:
            length += 1
        longest = max(longest, length)
    return longest


def isPalindrome(s: str) -> bool:
    res = []
    for char in s:
        if char.isalnum():
            res.append(char.lower())
    l, r = 0, len(res) - 1
    while l < r:
        if res[l] != res[r]:
            return False
        l += 1
        r -= 1
    return True
