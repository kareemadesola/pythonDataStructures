import collections
from typing import List


def distributeCookies(cookies: List[int], k: int) -> int:
    cur = [0] * k
    n = len(cookies)

    def dfs(i, zero_count):
        if n - i < zero_count:
            return float('inf')
        if i == n:
            return max(cur)

        res = float('inf')
        for j in range(k):
            zero_count -= int(cur[j] == 0)
            cur[j] += cookies[i]

            res = min(res, dfs(i + 1, zero_count))

            cur[j] -= cookies[i]
            zero_count += int(cur[j] == 0)
        return res

    return dfs(0, k)


def buddyStrings(s: str, goal: str) -> bool:
    if len(s) != len(goal):
        return False
    if sorted(s) != sorted(goal):
        return False
    if s == goal and len(set(s)) < len(goal):
        return True
    dif = [(i, j) for i, j in zip(s, goal) if i != j]
    return len(dif) == 2


def singleNumber(nums: List[int]) -> int:
    cnt = collections.Counter(nums)
    for k in cnt:
        if cnt[k] == 1: return k


def longestSubarray(nums: List[int]) -> int:
    k = 1
    i = 0
    for j in range(len(nums)):
        k -= nums[j] == 0
        if k < 0:
            k += nums[i] == 0
            i += 1
    return j - i
