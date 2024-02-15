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
            dp[start % K] = max(
                dp[start % K], dp[(i + 1) % K] + currMax * (i - start + 1)
            )

    return dp[0]


def firstUniqChar(s: str) -> int:
    seen = {}
    for idx, char in enumerate(s):
        if char in seen:
            seen[char] = float("inf")
        else:
            seen[char] = idx
    res = min(seen.values())
    return res if res != float("inf") else -1


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


def frequencySort(s: str) -> str:
    cnt = Counter(s)
    cnt_to_char = defaultdict(list)
    for char, count in cnt.items():
        cnt_to_char[count].append(char)
    res = []
    for count in range(len(s), -1, -1):
        for char in cnt_to_char[count]:
            res.append(char * count)
    return "".join(res)


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


def largestDivisibleSubset(nums: List[int]) -> List[int]:
    nums.sort()
    n = len(nums)
    memo = {}

    def dp(i: int, prev: int):
        if i == n:
            return []
        if (i, prev) in memo:
            return memo[(i, prev)]
        res = dp(i + 1, prev)
        if nums[i] % prev == 0:
            tmp = [nums[i]] + dp(i + 1, nums[i])
            res = tmp if len(tmp) > len(res) else res
        memo[(i, prev)] = res
        return res

    return dp(0, 1)


def largestDivisibleSubsetAlt(nums: List[int]) -> List[int]:
    nums.sort()
    memo = {}
    n = len(nums)

    def dp(i: int) -> List[int]:
        if i == n:
            return []
        if i in memo:
            return memo[i]
        res = [nums[i]]
        for j in range(i + 1, n):
            if nums[j] % nums[i] == 0:
                tmp = [nums[i]] + dp(j)
                res = tmp if len(tmp) > len(res) else res
        memo[i] = res
        return res

    ans = []
    for i in range(n):
        temp = dp(i)
        ans = temp if len(temp) > len(ans) else ans
    return ans


def largestDivisibleSubsetOpt(nums: List[int]) -> List[int]:
    nums.sort()
    n = len(nums)
    dp = [[val] for val in nums]
    res = []
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if nums[j] % nums[i] == 0 and len(dp[j]) >= len(dp[i]):
                dp[i] = [nums[i]] + dp[j]
        res = dp[i] if len(dp[i]) > len(res) else res
    return res


def countSubstrings(s: str) -> int:
    def is_palindrome(l: int, r: int):
        ans = 0
        while l >= 0 and r < n and s[l] == s[r]:
            ans += 1
            l -= 1
            r += 1
        return ans

    n = len(s)
    res = 0
    for i in range(n):
        res += is_palindrome(i, i) + is_palindrome(i, i + 1)
    return res


def cherryPickup(grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    memo = {}
    d = (-1, 0, 1)

    def dfs(r: int, c1: int, c2: int) -> int:
        if (r, c1, c2) in memo:
            return memo[(r, c1, c2)]
        if c1 == c2 or min(c1, c2) < 0 or max(c1, c2) == n or r == m:
            return 0
        res = 0
        for c1_d in d:
            for c2_d in d:
                res = max(res, dfs(r + 1, c1 + c1_d, c2 + c2_d))
        res += grid[r][c1] + grid[r][c2]
        memo[(r, c1, c2)] = res
        return res

    return dfs(0, 0, n - 1)


def cherryPickupAlt(grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    prev = [[0] * n for _ in range(n)]
    d = (-1, 0, 1)

    for r in range(m - 1, -1, -1):
        curr = [[0] * n for _ in range(n)]
        for c1 in range(n - 1):
            for c2 in range(c1 + 1, n):
                res = 0
                for c1_d in d:
                    for c2_d in d:
                        n_c1, n_c2 = c1 + c1_d, c2 + c2_d
                        if n_c1 < 0 or n_c2 == n:
                            continue
                        res = max(res, grid[r][c1] + grid[r][c2] + prev[n_c1][n_c2])
                curr[c1][c2] = res
        prev = curr
    return prev[0][n - 1]


def majorityElement(nums: List[int]) -> int:
    cnt = Counter(nums)
    n = len(nums)
    for val, count in cnt.items():
        if count > n // 2:
            return val


def majorityElementAlt(nums: List[int]) -> int:
    res = count = 0
    for val in nums:
        if count == 0:
            res = val
        count += 1 if res == val else -1
    return res


def rearrangeArray(nums: List[int]) -> List[int]:
    p_idx, n_idx = 0, 1
    res = [0] * len(nums)
    for val in nums:
        if val > 0:
            res[p_idx] = val
            p_idx += 2
        else:
            res[n_idx] = val
            n_idx += 2
    return res
