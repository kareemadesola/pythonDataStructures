from collections import Counter, defaultdict
from typing import List


def divideArray(nums: List[int], k: int) -> List[List[int]]:
    nums.sort()
    res = []
    for i in range(0, len(nums), 3):
        tmp = [nums[i], nums[i + 1], nums[i + 2]]
        if nums[i + 2] - nums[i] > k:
            return []
        res.append(tmp)
    return res


def maxSumAfterPartitioning(arr, k):
    N = len(arr)
    K = k + 1

    dp = [0] * K

    for start in range(N - 1, -1, -1):
        currMax = 0
        end = min(N, start + k)

        for i in range(start, end):
            currMax = max(currMax, arr[i])
            dp[start % K] = max(dp[start % K], dp[(i + 1) % K] + currMax * (i - start + 1))

    return dp[0]


def firstUniqChar(s: str) -> int:
    seen = {}
    for idx, char in enumerate(s):
        if char in seen:
            seen[char] = float('inf')
        else:
            seen[char] = idx
    res = min(seen.values())
    return res if res != float('inf') else -1


def firstUniqCharAlt(s: str) -> int:
    cnt = Counter(s)
    for idx, val in enumerate(s):
        if cnt[val] == 1:
            return idx
    return -1


def groupAnagrams(strs: List[str]) -> List[List[str]]:
    res = defaultdict(list)
    for val in strs:
        res[tuple(sorted(val))].append(val)
    return res.values()


def numSquares(n: int) -> int:
    dp = [n] * (n + 1)
    dp[0] = 0
    for remain in range(1, n + 1):
        for s in range(1, remain + 1):
            square = s * s
            if remain - square < 0:
                break
            dp[remain] = min(dp[remain], 1 + dp[remain - square])
    return dp[n]
