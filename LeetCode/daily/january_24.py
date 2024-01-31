import bisect
import collections
import heapq
import math
import random
from bisect import bisect_left
from collections import defaultdict, Counter
from math import ceil
from typing import List, Optional

from LeetCode.tree_visualizer import TreeNode


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


def numberOfArithmeticSlices(nums: List[int]) -> int:
    n = len(nums)
    dp = [defaultdict(int) for _ in range(n)]
    res = 0

    for i in range(n):
        for j in range(i):
            cnt = 0
            diff = nums[i] - nums[j]
            if diff in dp[j]:
                cnt = dp[j][diff]
            dp[i][diff] += cnt + 1
            res += cnt
    return res


def rangeSumBST(root: Optional[TreeNode], low: int, high: int) -> int:
    res = 0

    def dfs(node: TreeNode):
        nonlocal res
        if not node:
            return
        if low <= node.val <= high:
            res += node.val
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return res


def rangeSumBSTAlt(root: Optional[TreeNode], low: int, high: int) -> int:
    def dfs(node: TreeNode) -> int:
        if not node:
            return 0
        if low <= node.val <= high:
            return dfs(node.left) + node.val + dfs(node.right)
        elif node.val > high:
            return dfs(node.left)
        else:
            return dfs(node.right)

    return dfs(root)


def rangeSumBSTOP(root: Optional[TreeNode], low: int, high: int) -> int:
    res = 0

    def dfs(node: TreeNode):
        nonlocal res
        if not node:
            return
        if low <= node.val <= high:
            res += node.val
        if low < node.val:
            dfs(node.left)
        if node.val < high:
            dfs(node.right)

    dfs(root)
    return res


def leafSimilar(root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
    root1_list = []
    root2_list = []

    def closure(lst: List[int], root: Optional[TreeNode]) -> List[int]:
        res = lst

        def dfs(node: Optional[TreeNode]):
            if not node:
                return
            if not node.left and not node.right:
                res.append(node.val)
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return res

    return closure(root1_list, root1) == closure(root2_list, root2)


def leafSimilarAlt(root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
    root1_list = []
    root2_list = []

    def dfs(node: Optional[TreeNode], leaf_list: List[int]):
        if not node:
            return
        if not node.left and not node.right:
            leaf_list.append(node.val)
        dfs(node.left, leaf_list)
        dfs(node.right, leaf_list)

    dfs(root1, root1_list)
    dfs(root2, root2_list)
    return root1_list == root2_list


def halvesAreAlike(s: str) -> bool:
    n = len(s)
    vowels = ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
    res = 0
    for char in s[:n // 2]:
        if char in vowels:
            res += 1
    for char in s[n // 2:]:
        if char in vowels:
            res -= 1
    return not res


def halvesAreAlikeAlt(s: str) -> bool:
    n = len(s)
    vowels = ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')

    def count_vowels(start: int, end: int):
        res = 0
        for i in range(start, end):
            if s[i] in vowels:
                res += 1
        return res

    return count_vowels(0, n // 2) == count_vowels(n // 2, n)


def maxAncestorDiff(root: Optional[TreeNode]) -> int:
    res = 0

    def dfs(node: Optional[TreeNode], max_m: int, min_m: int):
        nonlocal res
        if not node:
            return
        new_max = max(max_m, node.val)
        new_min = min(min_m, node.val)
        res = max(res, new_max - new_min)
        dfs(node.left, new_max, new_min)
        dfs(node.right, new_max, new_min)

    dfs(root, root.val, root.val)
    return res


def maxAncestorDiffAlt(root: Optional[TreeNode]) -> int:
    def dfs(node: Optional[TreeNode], max_m: int, min_m: int) -> int:
        if not node:
            return max_m - min_m
        new_max = max(max_m, node.val)
        new_min = min(min_m, node.val)
        left = dfs(node.left, new_max, new_min)
        right = dfs(node.right, new_max, new_min)
        return max(left, right)

    return dfs(root, root.val, root.val)


def amountOfTime(root: Optional[TreeNode], start: int) -> int:
    adj_list = defaultdict(list)

    def tree_to_graph(node: Optional[TreeNode], parent: int):
        if not node:
            return
        if parent != 0:
            adj_list[node.val].append(parent)
        if node.left:
            adj_list[node.val].append(node.left.val)
        if node.right:
            adj_list[node.val].append(node.right.val)
        tree_to_graph(node.left, node.val)
        tree_to_graph(node.right, node.val)

    tree_to_graph(root, 0)
    q = collections.deque([start])
    seen = {start}
    res = 0
    while q:
        q_len = len(q)
        for _ in range(q_len):
            curr = q.popleft()
            for nei in adj_list[curr]:
                if nei not in seen:
                    seen.add(nei)
                    q.append(nei)
        res += 1
    return res - 1


def amountOfTimeAlt(root: Optional[TreeNode], start: int) -> int:
    max_distance = 0

    def dfs(node: Optional[TreeNode]) -> int:
        nonlocal max_distance
        depth = 0
        if not node:
            return depth
        left = dfs(node.left)
        right = dfs(node.right)

        if node.val == start:
            max_distance = max(left, right)
            depth = -1
        elif left >= 0 and right >= 0:
            depth = max(left, right) + 1
        else:
            distance = abs(left) + abs(right)
            max_distance = max(max_distance, distance)
            depth = min(left, right) - 1
        return depth

    dfs(root)
    return max_distance


def minSteps(self, s: str, t: str) -> int:
    s_cnt = Counter(s)
    res = 0
    for char in t:
        if s_cnt[char] > 0:
            s_cnt[char] -= 1
        else:
            res += 1
    return res


def minSteps(self, s: str, t: str) -> int:
    t_cnt = Counter(t)
    t_cnt.subtract(s)
    res = 0
    for _, val in t_cnt.items():
        if val > 0:
            res += val
    return res


def closeStrings(word1: str, word2: str) -> bool:
    word1_cnt = Counter(word1)
    word2_cnt = Counter(word2)
    return word1_cnt.keys() == word2_cnt.keys() and sorted(word1_cnt.values()) == sorted(
        word2_cnt.values())


def findWinners(matches: List[List[int]]) -> List[List[int]]:
    winner_set = set()
    loser_dict = defaultdict(int)
    for winner, loser in matches:
        winner_set.add(winner)
        loser_dict[loser] += 1

    res = [[], []]
    for winner in winner_set:
        if winner not in loser_dict:
            res[0].append(winner)

    res[0] = sorted(res[0])
    res[1] = sorted({loser for loser in loser_dict if loser_dict[loser] == 1})
    return res


def findWinnersAlt(matches: List[List[int]]) -> List[List[int]]:
    losses_cnt = {}
    for winner, loser in matches:
        losses_cnt.setdefault(winner, 0)
        losses_cnt[loser] = losses_cnt.get(loser, 0) + 1
    zero_losses = []
    one_loss = []
    for player, count in losses_cnt.items():
        if count == 0:
            zero_losses.append(player)
        elif count == 1:
            one_loss.append(player)
    return [sorted(zero_losses), sorted(one_loss)]


class RandomizedSet:

    def __init__(self):
        self.array = []
        self.hash_map = {}

    def insert(self, val: int) -> bool:
        if val in self.hash_map:
            return False
        self.hash_map[val] = len(self.array)
        self.array.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.hash_map:
            return False
        index = self.hash_map[val]
        last_element = self.array[-1]
        self.array[index] = last_element
        self.hash_map[last_element] = index

        self.array.pop()
        del self.hash_map[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self.array)


def uniqueOccurrences(arr: List[int]) -> bool:
    cnt = Counter(arr)
    return len(cnt.values()) == len(set(cnt.values()))


def climbStairs(n: int) -> int:
    if n == 1:
        return 1
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


def climbStairsAlt(n: int) -> int:
    if n == 1:
        return 1
    first, second = 1, 2
    for i in range(3, n + 1):
        first, second = second, first + second
    return second


def minFallingPathSum(matrix: List[List[int]]) -> int:
    N = len(matrix)
    memo = {}

    def dp(r: int, c: int) -> int:
        if r == N: return 0
        if c < 0 or c == N: return float('inf')
        if (r, c) in memo: return memo[(r, c)]
        res = matrix[r][c] + min(dp(r + 1, c - 1), dp(r + 1, c), dp(r + 1, c + 1))
        memo[(r, c)] = res
        return res

    return min(dp(0, col) for col in range(N))


def minFallingPathSumBU(matrix: List[List[int]]) -> int:
    n = len(matrix)
    dp = [num for num in matrix[-1]]
    for r in range(n - 2, -1, -1):
        new_dp = [0] * n
        for c in range(n):
            left = dp[c - 1] if c > 0 else float('inf')
            mid = dp[c]
            right = dp[c + 1] if c < n - 1 else float('inf')
            new_dp[c] = matrix[r][c] + min(left, mid, right)
        dp = new_dp

    return min(dp)


def minFallingPathSumOpt(matrix: List[List[int]]) -> int:
    N = len(matrix)
    for r in range(1, N):
        for c in range(N):
            left = matrix[r - 1][c - 1] if c > 0 else float('inf')
            mid = matrix[r - 1][c]
            right = matrix[r - 1][c + 1] if c < N - 1 else float('inf')
            matrix[r][c] += min(left, mid, right)
    return min(matrix[-1])


def findErrorNums(nums: List[int]) -> List[int]:
    cnt = Counter(nums)
    res = [-1, -1]
    n = len(nums)
    for i in range(1, n + 1):
        if cnt[i] == 0:
            res[1] = i
        elif cnt[i] == 2:
            res[0] = i
    return res


def findErrorNumsAlt(nums: List[int]) -> List[int]:
    res = [-1, -1]
    for n in nums:
        n = abs(n)
        nums[n - 1] = -nums[n - 1]
        if nums[n - 1] > 0:
            res[0] = n
    n = len(nums)
    for i in range(1, n + 1):
        if nums[i - 1] > 0 and i != res[0]:
            res[1] = i
            return res


def robAlt(nums: List[int]) -> int:
    n = len(nums)
    nxt = nxt_nxt = 0

    for i in range(n - 1, -1, -1):
        tmp = max(nums[i] + nxt_nxt, nxt)
        nxt, nxt_nxt = tmp, nxt
    return nxt


def rob(nums: List[int]) -> int:
    memo = {}
    n = len(nums)

    def dp(i: int) -> int:
        if i >= n:
            return 0
        if i in memo:
            return memo[i]
        memo[i] = max(nums[i] + dp(i + 2), dp(i + 1))
        return memo[i]

    return dp(0)


def maxLength(arr: List[str]) -> int:
    char_set = set()

    def overlap(c_set, s):
        c = Counter(c_set) + Counter(s)
        return max(c.values()) > 1

    # def overlap(c_set, s):
    #     return len(c_set) + len(s) != len(set(c_set).union(s))

    def backtrack(i: int) -> int:
        if i == len(arr):
            return len(char_set)
        res = 0

        if not overlap(char_set, arr[i]):
            for c in arr[i]:
                char_set.add(c)
            res = backtrack(i + 1)
            for c in arr[i]:
                char_set.remove(c)
        return max(res, backtrack(i + 1))

    return backtrack(0)


def pseudoPalindromicPaths(root: Optional[TreeNode]) -> int:
    cnt = defaultdict(int)
    odd = 0

    def dfs(curr: Optional[TreeNode]):
        nonlocal odd, res
        if not curr:
            return

        cnt[curr.val] += 1
        odd_change = 1 if cnt[curr.val] % 2 == 1 else -1
        odd += odd_change

        if not curr.left and not curr.right:
            res += 1 if odd <= 1 else 0
        else:
            dfs(curr.left)
            dfs(curr.right)
        odd -= odd_change
        cnt[curr.val] -= 1

    res = 0
    dfs(root)
    return res


def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if text1[i] == text2[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[0][0]


def longestCommonSubsequenceOpt(text1: str, text2: str) -> int:
    if len(text2) > len(text1):
        text1, text2 = text2, text1
    m, n = len(text1), len(text2)
    prev_dp = [0] * (n + 1)

    for i in range(m - 1, -1, -1):
        dp = [0] * (m + 1)
        for j in range(n - 1, -1, -1):
            if text1[i] == text2[j]:
                dp[j] = 1 + prev_dp[j + 1]
            else:
                dp[j] = max(dp[j + 1], prev_dp[j])
        prev_dp = dp
    return prev_dp[0]


def findPaths(m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
    memo = {}
    MOD = 10 ** 9 + 7

    def dfs(r: int, c: int, moves: int):
        if (r, c, moves) in memo:
            return memo[(r, c, moves)]
        if r < 0 or r == m or c < 0 or c == n:
            return 1
        if moves == 0:
            return 0
        res = (dfs(r - 1, c, moves - 1) + dfs(r + 1, c, moves - 1) + dfs(r, c - 1, moves - 1) + dfs(r, c + 1,
                                                                                                    moves - 1)) % MOD
        memo[(r, c, moves)] = res
        return res

    return dfs(startRow, startColumn, maxMove)


def findPathsAlt(m: int, n: int, N: int, x: int, y: int) -> int:
    M = 1000000000 + 7
    dp = [[0 for _ in range(n)] for _ in range(m)]
    dp[x][y] = 1
    count = 0

    for moves in range(1, N + 1):
        temp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i == m - 1:
                    count = (count + dp[i][j]) % M
                if j == n - 1:
                    count = (count + dp[i][j]) % M
                if i == 0:
                    count = (count + dp[i][j]) % M
                if j == 0:
                    count = (count + dp[i][j]) % M

                temp[i][j] = (
                                     ((dp[i - 1][j] if i > 0 else 0) + (dp[i + 1][j] if i < m - 1 else 0)) % M +
                                     ((dp[i][j - 1] if j > 0 else 0) + (dp[i][j + 1] if j < n - 1 else 0)) % M
                             ) % M
        dp = temp

    return count


def findPathsNC(m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
    MOD = 10 ** 9 + 7
    prev_dp = [[0] * n for _ in range(m)]
    for _ in range(1, maxMove + 1):
        dp = [[0] * n for _ in range(m)]
        for r in range(m):
            for c in range(n):
                if r - 1 < 0:
                    dp[r][c] = (dp[r][c] + 1) % MOD
                else:
                    dp[r][c] = (dp[r][c] + prev_dp[r - 1][c]) % MOD
                if r + 1 == m:
                    dp[r][c] = (dp[r][c] + 1) % MOD
                else:
                    dp[r][c] = (dp[r][c] + prev_dp[r + 1][c]) % MOD
                if c - 1 < 0:
                    dp[r][c] = (dp[r][c] + 1) % MOD
                else:
                    dp[r][c] = (dp[r][c] + prev_dp[r][c - 1]) % MOD
                if c + 1 == n:
                    dp[r][c] = (dp[r][c] + 1) % MOD
                else:
                    dp[r][c] = (dp[r][c] + prev_dp[r][c + 1]) % MOD
        prev_dp = dp
    return prev_dp[startRow][startColumn]


def kInversePairs(n: int, k: int) -> int:
    MOD = 10 ** 9 + 7
    prev = [0] * (k + 1)
    prev[0] = 1

    for N in range(1, n + 1):
        curr = [0] * (k + 1)
        total = 0
        for K in range(0, k + 1):
            if K >= N:
                total -= prev[K - N]
            total = (total + prev[K]) % MOD
            curr[K] = total
        prev = curr
    return prev[k]


def numSubmatrixSumTarget(matrix: List[List[int]], target: int) -> int:
    m, n = len(matrix), len(matrix[0])
    sub_sum = [[0] * n for _ in range(m)]

    for r in range(m):
        for c in range(n):
            top = sub_sum[r - 1][c] if r > 0 else 0
            left = sub_sum[r][c - 1] if c > 0 else 0
            top_left = sub_sum[r - 1][c - 1] if min(r, c) > 0 else 0
            sub_sum[r][c] = matrix[r][c] + top + left - top_left

    res = 0
    for r1 in range(m):
        for r2 in range(r1, m):
            cnt = defaultdict(int)
            cnt[0] = 1
            for c in range(n):
                curr_sum = sub_sum[r2][c] - (sub_sum[r1 - 1][c] if r1 > 0 else 0)
                diff = curr_sum - target
                res += cnt[diff]
                cnt[curr_sum] += 1
    return res


class MyQueue:

    def __init__(self):
        self.data = []

    def push(self, x: int) -> None:
        self.data.append(x)

    def pop(self) -> int:
        tmp = []
        n = len(self.data)
        for _ in range(n - 1):
            tmp.append(self.data.pop())
        res = self.data.pop()
        for _ in range(n - 1):
            self.data.append(tmp.pop())
        return res

    def peek(self) -> int:
        tmp = []
        n = len(self.data)
        for _ in range(n - 1):
            tmp.append(self.data.pop())
        res = self.data[-1]
        for _ in range(n - 1):
            self.data.append(tmp.pop())
        return res

    def empty(self) -> bool:
        return not self.data


class MyQueue2:

    def __init__(self):
        self.s1 = []
        self.s2 = []

    def push(self, x: int) -> None:
        self.s1.append(x)

    def pop(self) -> int:
        if not self.s2:
            self.move_s1_to_s2()
        return self.s2.pop()

    def peek(self) -> int:
        if not self.s2:
            self.move_s1_to_s2()
        return self.s2[-1]

    def empty(self) -> bool:
        return not (self.s1 or self.s2)

    def move_s1_to_s2(self):
        for _ in range(len(self.s1)):
            self.s2.append(self.s1.pop())


def evalRPN(tokens: List[str]) -> int:
    stack = []
    for t in tokens:
        if t in "+-*/":
            second = stack.pop()
            first = stack.pop()
            if t == "+":
                stack.append(first + second)
            elif t == "-":
                stack.append(first - second)
            elif t == "*":
                stack.append(first * second)
            elif t == "/":
                stack.append(math.trunc(first / second))
        else:
            stack.append(int(t))
    return stack[-1]


def dailyTemperatures(temperatures: List[int]) -> List[int]:
    stack = []
    res = [0] * len(temperatures)

    for i, t in enumerate(temperatures):
        while stack and t > temperatures[stack[-1]]:
            stack_idx = stack.pop()
            res[stack_idx] = (i - stack_idx)
        stack.append(i)
    return res
