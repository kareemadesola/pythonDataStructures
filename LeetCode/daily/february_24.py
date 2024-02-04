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
