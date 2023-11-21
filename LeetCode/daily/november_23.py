import collections
import heapq
from collections import defaultdict
from typing import List, Optional

from LeetCode.daily.october_23 import TreeNode


def findMode(root: Optional[TreeNode]) -> List[int]:
    def dfs(curr: TreeNode):
        val_to_cnt[curr.val] += 1
        if curr.left:
            dfs(curr.left)
        if curr.right:
            dfs(curr.right)

    val_to_cnt = defaultdict(int)
    dfs(root)
    mode = max(val_to_cnt.values())
    res = []
    for key, val in val_to_cnt.items():
        if val == mode:
            res.append(key)
    return res


def getWinner(arr: List[int], k: int) -> int:
    max_element = max(arr)
    curr = arr[0]
    win_streak = 0
    q = collections.deque(arr[1:])
    while q:
        opponent = q.popleft()
        if curr > opponent:
            win_streak += 1
            q.append(opponent)
        else:
            q.append(curr)
            curr = opponent
            win_streak = 1
        if win_streak == k or curr == max_element:
            return curr


def getWinnerAlt(arr: List[int], k: int) -> int:
    max_element = max(arr)
    curr = arr[0]
    win_streak = 0

    for i in range(1, len(arr)):
        opponent = arr[i]
        if curr > opponent:
            win_streak += 1
        else:
            curr = opponent
            win_streak = 1

        if win_streak == k or curr == max_element:
            return curr


class SeatManager:
    def __init__(self, n: int):
        self.available = [i for i in range(1, n + 1)]

    def reserve(self) -> int:
        return heapq.heappop(self.available)

    def unreserve(self, seatNumber: int) -> None:
        heapq.heappush(self.available, seatNumber)


def restoreArray(adjacentPairs: List[List[int]]) -> List[int]:
    adj_list = defaultdict(set)

    for x, y in adjacentPairs:
        adj_list[x].add(y)
        adj_list[y].add(x)

    root = None
    for curr in adj_list:
        if len(adj_list[curr]) == 1:
            root = curr
            break

    res = []

    def dfs(prev: int, curr: int):
        res.append(curr)
        for nei in adj_list[curr]:
            if nei != prev:
                dfs(curr, nei)

    dfs(None, root)
    return res


def restoreArrayAlt(adjacentPairs: List[List[int]]) -> List[int]:
    adj_list = defaultdict(list)

    for x, y in adjacentPairs:
        adj_list[x].append(y)
        adj_list[y].append(x)

    root = None
    for curr in adj_list:
        if len(adj_list[curr]) == 1:
            root = curr
            break

    res = [root]
    curr = root
    prev = None
    n = len(adj_list)

    while len(res) < n:
        for nei in adj_list[curr]:
            if nei != prev:
                res.append(nei)
                prev = curr
                curr = nei
                break
    return res


class Graph:

    def __init__(self, n: int, edges: List[List[int]]):
        self.n = n
        self.edges = defaultdict(list)
        for to, fro, weight in edges:
            self.edges[to].append((fro, weight))

    def addEdge(self, edge: List[int]) -> None:
        self.edges[edge[0]].append((edge[1], edge[2]))

    def shortestPath(self, node1: int, node2: int) -> int:
        seen = set()
        min_heap = [(0, node1)]
        while min_heap:
            weight, curr = heapq.heappop(min_heap)
            # if curr in seen:
            #     continue
            seen.add(curr)
            if curr == node2:
                return weight

            for nei_node, nei_weight in self.edges[curr]:
                if nei_node not in seen:
                    heapq.heappush(min_heap, (weight + nei_weight, nei_node))
        return -1


def sortVowels(s: str) -> str:
    vowels = set('aeiouAEIOU')
    n = len(s)
    res = [''] * n
    tmp = []
    for i in range(n):
        if s[i] not in vowels:
            res[i] = s[i]
        else:
            tmp.append(s[i])
    tmp.sort()
    for i in range(n - 1, -1, -1):
        if not res[i]:
            res[i] = tmp.pop()
    return ''.join(res)


def sortVowelsCountingSort(s: str) -> str:
    vowels = {'A': 0, 'E': 1, 'I': 2, 'O': 3, 'U': 4, 'a': 5, 'e': 6, 'i': 7, 'o': 8, 'u': 9}
    idx_to_vowel = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U', 5: 'a', 6: 'e', 7: 'i', 8: 'o', 9: 'u'}
    count = [0] * 10
    n = len(s)
    res = [''] * n
    for idx, val in enumerate(s):
        if val not in vowels:
            res[idx] = val
        else:
            count[vowels[val]] += 1

    j = 0
    for i in range(10):
        while count[i]:
            if not res[j]:
                res[j] = idx_to_vowel[i]
                count[i] -= 1
            j += 1

    return ''.join(res)


def maximumElementAfterDecrementingAndRearranging(arr: List[int]) -> int:
    arr.sort()
    n = len(arr)
    arr[0] = 1
    res = 1
    for i in range(1, n):
        res = min(res + 1, arr[i])
    return res


def maximumElementAfterDecrementingAndRearrangingAlt(arr: List[int]) -> int:
    n = len(arr)
    count = [0] * (n + 1)
    for num in arr:
        count[min(num, n)] += 1
    res = 1
    for i in range(2, n + 1):
        res = min(res + count[i], i)
    return res


def findDifferentBinaryString(nums: List[str]) -> str:
    n = len(nums)
    seen = set(nums)
    stack = []

    def backtrack(curr: int):
        if curr == n:
            string = ''.join(stack)
            return string if string not in seen else None
        for char in '01':
            stack.append(char)
            res = backtrack(curr + 1)
            if res: return res
            stack.pop()

    return backtrack(0)


def minPairSum(nums: List[int]) -> int:
    nums.sort()
    l, r = 0, len(nums) - 1
    res = 0
    while l < r:
        res = max(res, nums[l] + nums[r])
        l += 1
        r -= 1
    return max(res)


class GraphFloyd:

    def __init__(self, n: int, edges: List[List[int]]):
        self.cost = [[float('inf')] * n for _ in range(n)]
        for from_, to, edge_cost in edges:
            self.cost[from_][to] = edge_cost
        for i in range(n):
            self.cost[i][i] = 0
        for mid in range(n):
            for src in range(n):
                for dst in range(n):
                    self.cost[src][dst] = min(self.cost[src][dst], self.cost[src][mid] + self.cost[mid][dst])

    def addEdge(self, edge: List[int]) -> None:
        to, from_, weight = edge
        n = len(self.cost)
        for src in range(n):
            for dst in range(n):
                self.cost[src][dst] = min(self.cost[src][dst], self.cost[src][to] + self.cost[to][from_] + weight)

    def shortestPath(self, node1: int, node2: int) -> int:
        res = self.cost[node1][node2]
        return res if res != float('inf') else -1


def countNicePairs(nums: List[int]) -> int:
    def rev(val: int) -> int:
        return int(str(val)[::-1])

    for i in range(len(nums)):
        nums[i] = nums[i] - rev(nums[i])
    MOD = 10 ** 9 + 7
    cnt = defaultdict(int)
    res = 0
    for num in nums:
        res = (res + cnt[num]) % MOD
        cnt[num] += 1
    return res
