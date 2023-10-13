# Definition for a Node.
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
