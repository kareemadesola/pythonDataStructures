import bisect
import collections
import heapq
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
