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
