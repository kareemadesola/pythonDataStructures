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
            if curr in seen:
                continue
            seen.add(curr)
            if curr == node2:
                return weight

            for nei_node, nei_weight in self.edges[curr]:
                if nei_node not in seen:
                    heapq.heappush(min_heap, (weight + nei_weight, nei_node))
        return -1
