import bisect
import heapq
from bisect import bisect_left
from collections import defaultdict, Counter
from math import ceil
from typing import List


def findContentChildren(g: List[int], s: List[int]) -> int:
    g.sort()
    s.sort()
    content_children = 0
    for cookie in s:
        if content_children < len(g) and g[content_children] <= cookie:
            content_children += 1
    return content_children


def findMatrix(nums: List[int]) -> List[List[int]]:
    cnt = Counter(nums)
    res = []
    keys_to_delete = []
    while cnt:
        tmp = []
        for key, val in cnt.items():
            if val == 0:
                keys_to_delete.append(key)
            else:
                tmp.append(key)
                cnt[key] -= 1
        for key in keys_to_delete:
            del cnt[key]
        keys_to_delete.clear()
        if tmp:
            res.append(tmp)
    return res


def findMatrixAlt(nums: List[int]) -> List[List[int]]:
    cnt = defaultdict(int)
    res = []
    for n in nums:
        row = cnt[n]
        if len(res) == row:
            res.append([])
        res[row].append(n)
        cnt[n] += 1
    return res


def minOperations(nums: List[int]) -> int:
    cnt = Counter(nums)
    res = 0
    for item, count in cnt.items():
        if count == 1:
            return -1
        res += ceil(count / 3)
    return res


def minOperationsAlt(nums: List[int]) -> int:
    nums.sort()

    cache = {}

    def dfs(n):
        if n < 0:
            return float('inf')
        if n in (2, 3):
            return 1
        if n in cache:
            return cache[n]

        res = min(dfs(n - 2), dfs(n - 3))
        if res == -1:
            return -1
        cache[n] = res + 1
        return res + 1

    count = Counter(nums)
    res = 0
    for n, c in count.items():
        op = dfs(c)
        if op == float("inf"):
            return -1
        res += op
    return res


def lengthOfLIS(nums: List[int]) -> int:
    LIS = [1] * len(nums)
    n = len(nums)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if nums[j] > nums[i]:
                LIS[i] = max(LIS[i], 1 + LIS[j])
    return max(LIS)


def lengthOfLISAlt(nums: List[int]) -> int:
    LIS = [nums[0]]
    for num in nums:
        i = bisect_left(LIS, num)
        if i == len(LIS):
            LIS.append(num)
        else:
            LIS[i] = num
    return len(LIS)


def jobScheduling(startTime: List[int], endTime: List[int], profit: List[int]) -> int:
    interval = sorted(zip(startTime, endTime, profit))
    memo = {}

    def dp(i: int) -> int:
        if i == len(interval):
            return 0
        if i in memo: return memo[i]
        j = bisect.bisect(interval, (interval[i][1],))
        # max (don't include, include)
        memo[i] = max(dp(i + 1), interval[i][2] + dp(j))
        return memo[i]

    return dp(0)


def jobSchedulingBU(startTime: List[int], endTime: List[int], profit: List[int]) -> int:
    n = len(startTime)
    jobs = sorted(zip(startTime, endTime, profit))
    dp = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        j = bisect.bisect(jobs, (jobs[i][1],))
        dp[i] = max(dp[i + 1], jobs[i][2] + dp[j])
    return dp[0]


def jobSchedulingPQ(startTime: List[int], endTime: List[int], profit: List[int]) -> int:
    jobs = sorted(zip(startTime, endTime, profit))
    pq = []
    max_profit = 0
    for start, end, profit in jobs:
        while pq and start >= pq[0][0]:
            max_profit = max(max_profit, heapq.heappop(pq)[1])
        heapq.heappush(pq, (end, max_profit + profit))

    while pq:
        max_profit = max(max_profit, heapq.heappop(pq)[1])
    return max_profit
