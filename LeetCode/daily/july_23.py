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


def minSubArrayLen(target: int, nums: List[int]) -> int:
    start = 0
    tmp = 0
    res = len(nums) + 1
    for end in range(len(nums)):
        tmp += nums[end]
        while tmp >= target:
            res = min(res, end - start + 1)
            tmp -= nums[start]
            start += 1
    return res if res <= len(nums) else 0


def maxConsecutiveAnswers(answerKey: str, k: int) -> int:
    def check(chances, char) -> int:
        start = 0
        res = 0
        for end in range(len(answerKey)):
            if answerKey[end] == char:
                chances -= 1
                while chances < 0:
                    if answerKey[start] == char:
                        chances += 1
                    start += 1
            res = max(res, end - start + 1)
        return res

    return max(check(k, "F"), check(k, 'T'))


def putMarbles(weights: List[int], k: int) -> int:
    n = len(weights)
    pair_weights = [0] * (n - 1)
    for i in range(n - 1):
        pair_weights[i] = weights[i] + weights[i + 1]
    pair_weights.sort()

    res = 0
    for i in range(k - 1):
        res += pair_weights[n - 2 - i] - pair_weights[i]
    return res


def kadane_algorithm(nums) -> int:
    max_so_far = nums[0]
    max_ending_here = 0

    for num in nums:
        max_ending_here = max(max_ending_here, 0)
        max_ending_here += num
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far


def largestVariance(s: str) -> int:
    global_max = 0

    cnt = [0] * 26
    for i in s:
        cnt[ord(i) - ord('a')] += 1

    for i in range(26):
        major = chr(ord('a') + i)
        for j in range(26):

            if i == j or cnt[i] == 0 or cnt[j] == 0:
                continue

            minor = chr(ord('a') + j)
            major_count = 0
            minor_count = 0

            rest_minor = cnt[j]

            for char in s:
                if char == major:
                    major_count += 1
                if char == minor:
                    minor_count += 1
                    rest_minor -= 1

                if minor_count > 0:
                    global_max = max(global_max, major_count - minor_count)

                if minor_count > major_count and rest_minor:
                    major_count = 0
                    minor_count = 0

    return global_max
