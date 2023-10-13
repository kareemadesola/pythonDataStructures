# Definition for a Node.
from collections import deque
from typing import List


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


def preorder(root: Node) -> List[int]:
    res = []

    def dfs(curr: Node):
        if not curr:
            return
        res.append(curr.val)
        for child in curr.children:
            dfs(child)

    dfs(root)
    return res


def preorder_iter(root: Node) -> List[int]:
    res = []
    if not root:
        return res
    stack = [root]
    while stack:
        curr = stack.pop()
        res.append(curr.val)
        for child in curr.children[::-1]:
            stack.append(child)
    return res


def postorder(root: Node) -> List[int]:
    res = []
    if not root:
        return res

    def dfs(curr: Node):
        if not curr.children:
            res.append(curr.val)
        for child in curr.children:
            dfs(child)

    dfs(root)
    return res


def postorderIter(root: Node) -> List[int]:
    res = []
    if not root:
        return res

    stack = []
    while stack:
        curr = stack.pop()
        res.append(curr.val)
        for child in curr.children:
            stack.append(child)
    return res[::-1]


def levelOrder(root: Node) -> List[List[int]]:
    res = []
    if not root:
        return res
    q = deque([root])
    while q:
        curr_length = len(q)
        tmp = []
        for _ in range(curr_length):
            curr = q.popleft()
            tmp.append(curr.val)
            for child in curr.children:
                q.append(child)
        res.append(tmp)
    return res


def maxDepth(root: Node) -> int:
    if not root:
        return 0
    mx = 0

    def dfs(curr: Node, depth: int):
        nonlocal mx
        if not curr.children:
            mx = max(mx, depth)
        for child in curr.children:
            dfs(child, depth + 1)

    dfs(root, 1)
    return mx


def maxDepthBU(root: Node) -> int:
    if not root:
        return 0

    def dfs(curr: Node) -> int:
        if not curr.children:
            return 1
        return max(dfs(child) for child in curr.children) + 1

    return dfs(root)
