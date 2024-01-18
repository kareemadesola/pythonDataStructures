from collections import defaultdict
from typing import List


def findCircleNum(isConnected: List[List[int]]) -> int:
    n = len(isConnected)
    uf = UnionFind(n)
    for node_x in range(n):
        for node_y in range(node_x + 1, n):
            if isConnected[node_x][node_y]:
                uf.union(node_x, node_y)

    return uf.count


def validTree(n: int, edges: List[List[int]]) -> bool:
    if n - len(edges) != 1:
        return False
    uf = UnionFind(n)
    for node_x, node_y in edges:
        uf.union(node_x, node_y)
    return uf.count == 1


def countComponents(n: int, edges: List[List[int]]) -> int:
    uf = UnionFind(n)
    for node_x, node_y in edges:
        uf.union(node_x, node_y)
    return uf.count


def earliestAcq(logs: List[List[int]], n: int) -> int:
    logs.sort()
    uf = UnionFind(n)
    for timestamp, node_x, node_y in logs:
        uf.union(node_x, node_y)
        if uf.count == 1:
            return timestamp
    return -1


def smallestStringWithSwaps(s: str, pairs: List[List[int]]) -> str:
    n = len(s)
    uf = UnionFind(n)
    for node_x, node_y in pairs:
        uf.union(node_x, node_y)
    root_to_components = defaultdict(list)

    for vertex in range(n):
        root = uf.find(vertex)
        root_to_components[root].append(vertex)

    res = [''] * n
    for indices in root_to_components.values():
        characters = [s[index] for index in indices]
        characters.sort()

        for i, index in enumerate(indices):
            res[index] = characters[i]
    return ''.join(res)


class UnionFind:
    def __init__(self, size):
        self.root = [i for i in range(size)]
        self.rank = [1] * size
        self.count = size

    def find(self, x: int) -> int:
        if x == self.root[x]:
            return x
        self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x: int, y: int):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return

        if self.rank[root_x] > self.rank[root_y]:
            self.root[root_y] = root_x
        elif self.rank[root_x] < self.rank[root_y]:
            self.root[root_x] = root_y
        else:
            self.root[root_y] = root_x
            self.rank[root_x] += 1
        self.count -= 1


def calcEquation(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    gid_weight = {}

    def find(node_id):
        if node_id not in gid_weight:
            gid_weight[node_id] = (node_id, 1)
        group_id, node_weight = gid_weight[node_id]

        if group_id == node_id:
            return gid_weight[node_id]
        new_group_id, group_weight = find(group_id)
        gid_weight[node_id] = (new_group_id, node_weight * group_weight)
        return gid_weight[node_id]

    def union(dividend, divisor, value):
        dividend_gid, dividend_weight = find(dividend)
        divisor_gid, divisor_weight = find(divisor)

        if dividend_gid != divisor_gid:
            gid_weight[dividend_gid] = (divisor_gid, divisor_weight * value / dividend_weight)

    for (dividend, divisor), value in zip(equations, values):
        union(dividend, divisor, value)

    res = []
    for (dividend, divisor) in queries:
        if dividend not in gid_weight or divisor not in gid_weight:
            res.append(-1.0)
        else:
            dividend_gid, dividend_weight = find(dividend)
            divisor_gid, divisor_weight = find(divisor)

            if dividend_gid != divisor_gid:
                res.append(-1.0)
            else:
                res.append(dividend_weight / divisor_weight)
    return res


def minCostToSupplyWater(n: int, wells: List[int], pipes: List[List[int]]) -> int:
    root = [i for i in range(n + 1)]
    rank = [1] * (n + 1)

    def find(x: int):
        if x == root[x]:
            return x
        root[x] = find(root[x])
        return root[x]

    def union(x: int, y: int) -> bool:
        root_x = find(x)
        root_y = find(y)

        if root_x == root_y:
            return False
        if rank[root_x] > rank[root_y]:
            root[root_y] = root_x
        elif rank[root_x] < rank[root_y]:
            root[root_x] = root_y
        else:
            root[root_y] = root_x
            rank[root_x] += 1
        return True

    ordered_edges = []
    for index, weight in enumerate(wells):
        ordered_edges.append((weight, 0, index + 1))

    for house_1, house_2, weight in pipes:
        ordered_edges.append((weight, house_1, house_2))

    ordered_edges.sort(key=lambda x: x[0])

    total_cost = 0
    for cost, house_1, house_2 in ordered_edges:
        if union(house_1, house_2):
            total_cost += cost
    return total_cost


def validPath(n: int, edges: List[List[int]], source: int, destination: int) -> bool:
    adj_list = defaultdict(list)
    for src, dst in edges:
        adj_list[src].append(dst)
        adj_list[dst].append(src)

    def dfs(node: int) -> bool:
        if node == destination:
            return True

        for nei in adj_list[node]:
            if nei in visited:
                continue
            visited.add(nei)
            if dfs(nei):
                return True
        return False

    visited = {source}
    return dfs(source)


if __name__ == '__main__':
    n, source, destination = 10, 5, 9
    edges = [[4, 3], [1, 4], [4, 8], [1, 7], [6, 4], [4, 2], [7, 4], [4, 0], [0, 9], [5, 4]]
    validPath(n, edges, source, destination)

# def allPathsSourceTarget(graph: List[List[int]]) -> List[List[int]]:
#     target = len(graph) - 1
#
#     def backtrack(curr: int):
#         if curr == target:
#             res.append(path[:])
#             return
#         for nei in graph[curr]:
#             path.append(nei)
#             backtrack(nei)
#             path.pop()
#
#     res = []
#     path = [0]
#     backtrack(0)
#     return res


# Definition for a Node.
# class Node:
#     def __init__(self, val=0, neighbors=None):
#         self.val = val
#         self.neighbors = neighbors if neighbors else []
#
#
# def cloneGraph(node: Optional['Node']) -> Optional['Node']:
#     def clone(curr: Optional['Node']) -> Optional['Node']:
#         pass
#
#     return clone(node)
