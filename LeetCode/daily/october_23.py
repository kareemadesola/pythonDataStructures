import bisect
from collections import Counter, defaultdict
from math import comb
from typing import List


def reverseWords(s: str) -> str:
    return " ".join(word[::-1] for word in s.split(" "))


def winnerOfGame(colors: str) -> bool:
    global_cnt_a = global_cnt_b = 0
    local_cnt_a = local_cnt_b = 0

    for char in colors:
        if char == "A":
            local_cnt_a += 1
        else:
            global_cnt_a += max(local_cnt_a - 2, 0)
            local_cnt_a = 0
    global_cnt_a += max(local_cnt_a - 2, 0)

    for char in colors:
        if char == "B":
            local_cnt_b += 1
        else:
            global_cnt_b += max(local_cnt_b - 2, 0)
            local_cnt_b = 0
    global_cnt_b += max(local_cnt_b - 2, 0)
    return global_cnt_a > global_cnt_b


def winnerOfGameOptimal(colors: str) -> bool:
    alice = bob = 0
    for i in range(1, len(colors) - 1):
        if colors[i - 1] == colors[i] == colors[i + 1]:
            if colors[i] == "A":
                alice += 1
            else:
                bob += 1
    return alice > bob


def numIdenticalPairs(nums: List[int]) -> int:
    counter = Counter(nums)
    res = 0
    for count in counter.values():
        res += comb(count, 2)
    return res


class MyHashMap:
    def __init__(self):
        self.data = {}

    def put(self, key: int, value: int) -> None:
        self.data[key] = value

    def get(self, key: int) -> int:
        return self.data.get(key, -1)

    def remove(self, key: int) -> None:
        if key in self.data:
            del self.data[key]


class ListNode:
    def __init__(self, key=-1, val=-1, nxt=None):
        self.key = key
        self.val = val
        self.next = nxt


class MyHashMapBetter:
    def __init__(self):
        self.data = [ListNode() for _ in range(1000)]

    def hash(self, key):
        return key % 1000

    def put(self, key: int, value: int) -> None:
        curr = self.data[self.hash(key)]
        while curr and curr.next:
            if curr.next.key == key:
                curr.next.val = value
                return
            curr = curr.next
        curr.next = ListNode(key, value)

    def get(self, key: int) -> int:
        curr = self.data[self.hash(key)]
        while curr:
            if curr.key == key:
                return curr.val
            curr = curr.next
        return -1

    def remove(self, key: int) -> None:
        curr = self.data[self.hash(key)]
        while curr and curr.next:
            if curr.next.key == key:
                curr.next = curr.next.next
                return
            curr = curr.next


def majorityElement(nums: List[int]) -> List[int]:
    nums_len = len(nums)
    elem_to_freq = Counter(nums)
    res = []
    for elem in elem_to_freq:
        if elem_to_freq[elem] > nums_len // 3:
            res.append(elem)
    return res


def majorityElementConstantSpace(nums: List[int]) -> List[int]:
    count = defaultdict(int)

    for num in nums:
        count[num] += 1

        if len(count) < 3:
            continue

        new_count = defaultdict(int)
        for k, v in count.items():
            if count[k] > 1:
                new_count[k] = v - 1
    n = len(nums) // 3
    return [k for k in count if nums.count(k) > n]


def integerBreakRecursive(n: int) -> int:
    memo = {1: 1}

    def dfs(val: int) -> int:
        if val in memo:
            return memo[val]
        memo[val] = 0 if val == n else val
        for i in range(1, val):
            res = dfs(i) * dfs(val - i)
            memo[val] = max(memo[val], res)
        return memo[val]

    return dfs(n)


def integerBreak(n: int) -> int:
    dp = [i for i in range(n + 1)]
    dp[n] = 0

    for val in range(2, n + 1):
        for i in range(1, val):
            res = dp[i] * dp[val - i]
            dp[val] = max(dp[val], res)
    return dp[n]


def maxDotProduct(nums1: List[int], nums2: List[int]) -> int:
    memo = {}

    def dfs(i: int, j: int):
        if (i, j) in memo:
            return memo[(i, j)]
        if i == len(nums1) or j == len(nums2):
            return 0

        res = max(nums1[i] * nums2[j] + dfs(i + 1, j + 1), dfs(i + 1, j), dfs(i, j + 1))
        memo[(i, j)] = res
        return memo[(i, j)]

    nums1_min, nums1_max = min(nums1), max(nums1)
    nums2_min, nums2_max = min(nums2), max(nums2)

    if nums1_max < 0 and nums2_min > 0:
        return nums1_max * nums2_min

    if nums2_max < 0 and nums1_min > 0:
        return nums2_max * nums1_min

    return dfs(0, 0)


def maxDotProductBU(nums1: List[int], nums2: List[int]) -> int:
    nums1_min, nums1_max = min(nums1), max(nums1)
    nums2_min, nums2_max = min(nums2), max(nums2)

    if nums1_max < 0 and nums2_min > 0:
        return nums1_max * nums2_min

    if nums2_max < 0 and nums1_min > 0:
        return nums2_max * nums1_min
    m, n = len(nums1), len(nums2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m, -1, -1):
        for j in range(n, -1, -1):
            dp[i][j] = max(
                nums1[i] * nums2[j] + dp[i + 1][j + 1], dp[i + 1][j], dp[i][j + 1]
            )
    return dp[0][0]


def maxDotProductSOBU(nums1: List[int], nums2: List[int]) -> int:
    nums1_min, nums1_max = min(nums1), max(nums1)
    nums2_min, nums2_max = min(nums2), max(nums2)

    if nums1_max < 0 and nums2_min > 0:
        return nums1_max * nums2_min

    if nums2_max < 0 and nums1_min > 0:
        return nums2_max * nums1_min

    m, n = len(nums1), len(nums2)
    prev_dp = [0] * (n + 1)

    for i in range(m - 1, -1, -1):
        dp = [0] * (n + 1)
        for j in range(n - 1, -1, -1):
            dp[j] = max(nums1[i] * nums2[j] + prev_dp[j + 1], prev_dp[j], dp[j + 1])
        prev_dp = dp
    return prev_dp[0]


def searchRange(nums: List[int], target: int) -> List[int]:
    l = bisect.bisect_left(nums, target)
    r = bisect.bisect_right(nums, target)
    return [l, r - 1] if l != r else [-1, -1]
