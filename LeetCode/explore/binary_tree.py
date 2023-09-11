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
