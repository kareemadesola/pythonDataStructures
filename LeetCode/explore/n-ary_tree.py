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
    q = deque([root])
    while q:
        curr = q.popleft()
        res.append(curr.val)
        tmp = []
        for child in curr.children:
            tmp.append(child)
        q.extendleft(tmp[::-1])
    return res
