import bisect
import collections
import heapq
import math
from collections import Counter, defaultdict
from math import comb
from typing import List, Optional


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


def searchRangeBisect(nums: List[int], target: int) -> List[int]:
    l = bisect.bisect_left(nums, target)
    r = bisect.bisect_right(nums, target)
    return [l, r - 1] if l != r else [-1, -1]


def searchRange(nums: List[int], target: int) -> List[int]:
    def bisect_left():
        l, r = 0, len(nums)
        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] < target:
                l = mid + 1
            else:
                r = mid
        return l

    def bisect_right():
        l, r = 0, len(nums)
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] < target:
                l = mid + 1
            else:
                r = mid
        return l

    left, right = bisect_left(), bisect_right()
    return [left, right - 1] if left != right else [-1, -1]


def minOperations(nums: List[int]) -> int:
    length = len(nums)
    nums = sorted(set(nums))
    res = length
    r = 0

    for l in range(len(nums)):
        while r < len(nums) and nums[r] < nums[l] + length:
            r += 1
        window = r - l
        res = min(res, length - window)
    return res


def minOperationsBS(nums: List[int]) -> int:
    n = len(nums)
    nums = sorted(set(nums))
    res = n

    for l in range(len(nums)):
        r = bisect.bisect(nums, nums[l] + n - 1)
        window = r - l
        res = min(res, n - window)
    return res


def fullBloomFlowers(flowers: List[List[int]], people: List[int]) -> List[int]:
    people = [(p, i) for i, p in enumerate(people)]
    res = [0] * len(people)
    flowers.sort()
    heap = []

    j = 0
    for p, i in sorted(people):
        while j < len(flowers) and p >= flowers[j][0]:
            heapq.heappush(heap, flowers[j][1])
            j += 1
        while heap and p > heap[0]:
            heapq.heappop(heap)
        res[i] = len(heap)
    return res


class MountainArray:
    def get(self, index: int) -> int:
        pass

    def length(self) -> int:
        pass


def findInMountainArray(target: int, mountain_arr: "MountainArray") -> int:
    # get index of peak
    def find_peak(l: int, r: int):
        while l < r:
            mid = l + (r - l) // 2
            if mountain_arr.get(mid) < mountain_arr.get(mid + 1):
                l = mid + 1
            else:
                r = mid
        return l

    def binary_search_left(l: int, r: int):
        while l <= r:
            mid = l + (r - l) // 2
            if mountain_arr.get(mid) == target:
                return mid
            if mountain_arr.get(mid) < target:
                l = mid + 1
            else:
                r = mid - 1
        return -1

    def binary_search_right(l: int, r: int):
        while l <= r:
            mid = l + (r - l) // 2
            if mountain_arr.get(mid) == target:
                return mid
            if mountain_arr.get(mid) < target:
                r = mid - 1
            else:
                l = mid + 1
        return -1

    end = mountain_arr.length() - 1
    peak = find_peak(0, end)

    l_section = binary_search_left(0, peak)
    return l_section if l_section != -1 else binary_search_right(peak + 1, end)


def minCostClimbingStairs(cost: List[int]) -> int:
    n = len(cost)
    dp = [0] * n
    dp[0], dp[1] = cost[0], cost[1]
    for i in range(2, n):
        dp[i] = cost[i] + min(dp[i - 1], dp[i - 2])

    return min(dp[n - 1], dp[n - 2])


def minCostClimbingStairsOp(cost: List[int]) -> int:
    n = len(cost)
    prev_prev, prev = cost[0], cost[1]

    for i in range(2, n):
        dp = cost[i] + min(prev, prev_prev)
        prev_prev, prev = prev, dp

    return min(prev, prev_prev)


def minCostClimbingStairsTD(cost: List[int]) -> int:
    memo = {0: cost[0], 1: cost[1]}
    n = len(cost)

    def dfs(i: int) -> int:
        if i == n:
            return min(dfs(i - 1), dfs(i - 2))
        if i in memo:
            return memo[i]
        memo[i] = cost[i] + min(dfs(i - 1), dfs(i - 2))
        return memo[i]

    return dfs(n)


def paintWalls(cost: List[int], time: List[int]) -> int:
    memo = {}

    def dfs(i: int, remains: int) -> int | float:
        if remains <= 0:
            return 0
        if i == len(cost):
            return float("inf")
        if (i, remains) in memo:
            return memo[(i, remains)]
        paint = cost[i] + dfs(i + 1, remains - 1 - time[i])
        skip = dfs(i + 1, remains)
        memo[(i, remains)] = min(paint, skip)
        return memo[(i, remains)]

    return dfs(0, len(cost))


def paintWallsBU(cost: List[int], time: List[int]) -> int:
    n = len(cost)
    dp: List[List[int | float]] = [[0] * (n + 1) for _ in range(n + 1)]

    # base case
    # i == n or remains == 0
    for remains in range(1, n + 1):
        dp[n][remains] = float("inf")

    for i in range(n - 1, -1, -1):
        for remains in range(1, n + 1):
            paint = cost[i] + dp[i + 1][max(0, remains - 1 - time[i])]
            skip = dp[i + 1][remains]
            dp[i][remains] = min(paint, skip)
    return dp[0][n]


def paintWallsSOBU(cost: List[int], time: List[int]) -> int:
    n = len(cost)
    # i == n
    prev_dp: List[int | float] = [float("inf")] * (n + 1)
    prev_dp[0] = 0

    for i in range(n - 1, -1, -1):
        dp = [0] * (n + 1)
        for remains in range(1, n + 1):
            paint = cost[i] + prev_dp[max(0, remains - 1 - time[i])]
            skip = prev_dp[remains]
            dp[remains] = min(paint, skip)
        prev_dp = dp
    return prev_dp[n]


def numWays(steps: int, arrLen: int) -> int:
    memo = {}
    MOD = 10**9 + 7

    def dfs(curr, remain) -> int:
        if remain == 0:
            return 1 if curr == 0 else 0
        if (curr, remain) in memo:
            return memo[(curr, remain)]
        res = dfs(curr, remain - 1) % MOD
        if curr > 0:
            res += dfs(curr - 1, remain - 1) % MOD
        if curr < arrLen - 1:
            res += dfs(curr + 1, remain - 1) % MOD
        memo[(curr, remain)] = res % MOD
        return memo[(curr, remain)]

    return dfs(0, steps)


def backspaceCompare(s: str, t: str) -> bool:
    stack = []
    for char in s:
        if char == "#":
            stack.pop()
        else:
            stack.append(char)

    t_stack = []
    for char in t:
        if char == "#":
            stack.pop()
        else:
            stack.append(char)
    return stack == t_stack


def backspaceCompareO1_space(s: str, t: str) -> bool:
    def next_valid_char(string, idx):
        backspace = 0
        while idx >= 0:
            if string[idx] != "#":
                if backspace == 0:
                    break
                backspace -= 1
            else:
                backspace += 1
            idx -= 1
        return idx

    s_idx, t_idx = len(s) - 1, len(t) - 1
    while s_idx >= 0 or t_idx >= 0:
        s_idx = next_valid_char(s, s_idx)
        t_idx = next_valid_char(t, t_idx)

        s_char = s[s_idx] if s_idx >= 0 else ""
        t_char = t[t_idx] if t_idx >= 0 else ""

        if s_char != t_char:
            return False
        s_idx -= 1
        t_idx -= 1
    return True


def constrainedSubsetSum(nums: List[int], k: int) -> int:
    nums_len = len(nums)
    l = r = k
    res = nums[k]
    curr_min = nums[k]

    while l > 0 or r < nums_len - 1:
        left = nums[l - 1] if l > 0 else 0
        right = nums[r + 1] if r < nums_len - 1 else 0
        if left > right:
            l -= 1
            curr_min = min(curr_min, left)
        else:
            r += 1
            curr_min = min(curr_min, right)
        res = max(res, curr_min * (r - l + 1))
    return res


def isPowerOfFour(n: int) -> bool:
    return n > 0 and math.log(n, 4) % 1 == 0


def isPowerOfFourRecursive(n: int) -> bool:
    if n <= 0:
        return False

    def dfs(x) -> bool:
        if x == 1:
            return True
        return dfs(x // 4) if x % 4 == 0 else False

    return dfs(n)


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right


def largestValues(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    q = collections.deque([root])
    res = []
    while q:
        row_max = q[0].val
        q_len = len(q)
        for _ in range(q_len):
            curr = q.popleft()
            row_max = max(row_max, curr.val)
            if curr.left:
                q.append(curr.left)
            if curr.right:
                q.append(curr.right)
        res.append(row_max)
    return res


def kthGrammar(n: int, k: int) -> int:
    if n == 1:
        return 0
    parent = kthGrammar(n - 1, (k + 1) // 2)
    if parent:
        return 1 if k % 2 else 0
    return 0 if k % 2 else 1


def kthGrammarIter(n: int, k: int) -> int:
    l, r = 1, 2 ** (n - 1)
    curr = 0
    while l < r:
        mid = l + (r - l) // 2
        if mid < k:
            curr = 0 if curr else 1
            l = mid + 1
        else:
            r = mid
    return curr


def numFactoredBinaryTrees(arr: List[int]) -> int:
    dp = {}
    for val in sorted(arr):
        dp[val] = (
            sum(dp[elem] * dp.get(val // elem, 0) for elem in dp if val % elem == 0) + 1
        )
    return sum(dp.values()) % (10**9 + 7)


def longestPalindrome(s: str) -> str:
    s_len = len(s)

    def palindrome_len(start: int, end: int) -> tuple[int, int]:
        while start >= 0 and end < s_len:
            if s[start] == s[end]:
                start -= 1
                end += 1
            else:
                return start + 1, end - 1
        return start, end

    global_start = 0
    global_end = 0
    for i in range(s_len):
        local_start_1, local_end_1 = palindrome_len(i, i)
        local_start_2, local_end_2 = palindrome_len(i, i + 1)

        tmp_start, tmp_end = (
            (local_start_1, local_end_1)
            if (local_end_1 - local_start_1 > local_end_2 - local_start_2)
            else (local_start_2, local_end_2)
        )

        global_start, global_end = (
            (tmp_start, tmp_end)
            if (tmp_end - tmp_start) > (global_end - global_start)
            else (global_start, global_end)
        )
    return s[global_start : global_end + 1]


def longestPalindromeAlt(s: str) -> str:
    res = ""
    res_len = 0

    def palindrome(l: int, r: int):
        nonlocal res_len, res
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        if r - l - 1 > res_len:
            res = s[l + 1 : r]
            res_len = r - l - 1

    for i in range(len(s)):
        palindrome(i, i)
        palindrome(i, i + 1)

    return res
