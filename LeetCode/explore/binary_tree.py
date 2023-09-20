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


def isSymmetricBFS(root: Optional[TreeNode]) -> bool:
    q = collections.deque([root])
    while q:
        tmp = []
        q_len = len(q)
        for _ in range(q_len):
            curr = q.popleft()
            tmp.append(curr.val if curr else None)
            if not curr:
                continue
            q.append(curr.left)

            q.append(curr.right)
        tmp_len = len(tmp) // 2
        if tmp[:tmp_len] != tmp[-1 : -tmp_len - 1 : -1]:
            return False

    return True


def isSymmetricDFS(root: Optional[TreeNode]) -> bool:
    def dfs(left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (
            left.val == right.val
            and dfs(left.left, right.right)
            and dfs(left.right, right.left)
        )

    return dfs(root.left, root.right)


def hasPathSum(root: Optional[TreeNode], targetSum: int) -> bool:
    def dfs(curr: Optional[TreeNode], remainder: int) -> bool:
        if not curr:
            return False
        remainder -= curr.val
        if not curr.left and not curr.right:
            return remainder == curr.val
        return dfs(curr.left, remainder) or dfs(curr.right, remainder)

    return dfs(root, targetSum)


def buildTree(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    val_to_idx = {val: idx for idx, val in enumerate(inorder)}

    def dfs(l: int, r: int):
        if l > r:
            return
        root = TreeNode(postorder.pop())
        root_idx = val_to_idx[root.val]

        root.right = dfs(root_idx + 1, r)
        root.left = dfs(l, root_idx - 1)
        return root

    return dfs(0, len(inorder) - 1)


def buildTreePI(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    preorder = collections.deque(preorder)
    val_to_idx = {val: idx for idx, val in enumerate(inorder)}

    def dfs(l, r):
        if l > r:
            return
        root = TreeNode(preorder.popleft())
        root_idx = val_to_idx[root.val]

        root.left = dfs(l, root_idx - 1)
        root.right = dfs(root_idx + 1, r)
        return root

    return dfs(0, len(preorder) - 1)


class Node:
    def __init__(
        self,
        val: int = 0,
        left: "Node" = None,
        right: "Node" = None,
        next: "Node" = None,
    ):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


# def connect(root: "Optional[Node]") -> "Optional[Node]":
#     def dfs(node):
#         if not node:
#             return
#         if node.left:
#             node.next = node.right
#         return node
#
#     return dfs(root)


def connect(root: "Optional[Node]") -> "Optional[Node]":
    if not root:
        return

    q = collections.deque([root])
    while q:
        q_len = len(q)
        for i in range(q_len):
            curr = q.popleft()
            nxt = q[0] if i < q_len - 1 else None

            curr.next = nxt

            if curr.left:
                q.append(curr.left)
            if curr.right:
                q.append(curr.right)
    return root


def connectBetter(root: "Optional[Node]") -> "Optional[Node]":
    if not root:
        return
    curr, nxt = root, root.left

    while curr and nxt:
        curr.left.next = curr.right
        if curr.next:
            curr.right.next = curr.next.left
        curr = curr.next
        if not curr:
            curr = nxt
            nxt = curr.left
    return root


def lowestCommonAncestor(root: "TreeNode", p: "TreeNode", q: "TreeNode") -> "TreeNode":
    def dfs(node: TreeNode) -> Optional[TreeNode]:
        if not node:
            return
        if node == p or node == q:
            return node
        left = dfs(node.left)
        right = dfs(node.right)

        if left and right:
            return node
        if left:
            return left
        return right

    return dfs(root)
