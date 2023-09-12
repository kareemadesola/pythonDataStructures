import collections
from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def preorderTraversal(root: Optional[TreeNode]) -> List[int]:
    res = []

    def dfs(node: Optional[TreeNode]):
        if not node:
            return
        res.append(node.val)
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return res


def preorderTraversalIter(root: Optional[TreeNode]) -> List[int]:
    stack = [root]
    res = []

    while stack:
        node = stack.pop()
        if node:
            res.append(node.val)
            stack.append(node.right)
            stack.append(node.left)
    return res


def inorderTraversal(root: Optional[TreeNode]) -> List[int]:
    res = []

    def dfs(curr: Optional[TreeNode]):
        if not curr:
            return
        dfs(curr.left)
        res.append(curr.val)
        dfs(curr.right)

    dfs(root)
    return res


def inorderTraversalIter(root: Optional[TreeNode]) -> List[int]:
    stack = []
    res = []
    curr = root
    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        curr = curr.right
    return res


def postorderTraversal(root: Optional[TreeNode]) -> List[int]:
    def dfs(curr: Optional[TreeNode]):
        if not curr:
            return
        dfs(curr.left)
        dfs(curr.right)
        res.append(curr.val)

    res = []
    dfs(root)
    return res


def postorderTraversalIter(root: Optional[TreeNode]) -> List[int]:
    res = []
    stack = [root]
    while stack:
        curr = stack.pop()
        if curr:
            res.append(curr.val)
            stack.append(curr.left)
            stack.append(curr.right)
    return res[::-1]


def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    q = collections.deque([root])
    res = []
    while q:
        res.append([])
        curr_length = len(q)
        for _ in range(curr_length):
            curr = q.popleft()
            res[-1].append(curr.val)
            if curr.left:
                q.append(curr.left)
            if curr.right:
                q.append(curr.right)
    return res


def maxDepthTD(root: Optional[TreeNode]) -> int:
    def dfs(curr: Optional[TreeNode], depth: int):
        nonlocal res
        if not curr:
            return
        if not curr.left and not curr.right:
            res = max(res, depth)
        dfs(curr.left, depth + 1)
        dfs(curr.right, depth + 1)

    res = 0
    dfs(root, 1)
    return res


def maxDepthBU(root: Optional[TreeNode]) -> int:
    def dfs(curr: Optional[TreeNode]) -> int:
        if not curr:
            return 0
        left = dfs(curr.left)
        right = dfs(curr.right)
        return max(left, right) + 1

    return dfs(root)
