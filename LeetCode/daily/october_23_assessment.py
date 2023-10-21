# Definition for a binary tree node.
import collections
import math
from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right


def findTarget(root: Optional[TreeNode], k: int) -> bool:
    def convert_to_list() -> List[int]:
        res = []
        q = collections.deque([root])
        while q:
            q_len = len(q)
            for _ in range(q_len):
                curr = q.popleft()
                res.append(curr.val)
                if curr.left:
                    q.append(curr.left)
                if curr.right:
                    q.append(curr.right)
        return res

    def two_sum(arr: List[int]) -> bool:
        seen = set()
        for val in arr:
            if k - val in seen:
                return True
            seen.add(val)
        return False

    return two_sum(convert_to_list())


def findTargetDFS(root: Optional[TreeNode], k: int) -> bool:
    seen = set()

    def dfs(curr: TreeNode) -> bool:
        if not curr:
            return False
        if k - curr.val in seen:
            return True
        seen.add(curr.val)
        return dfs(curr.left) or dfs(curr.right)

    return dfs(root)


def findTargetBST(root: Optional[TreeNode], k: int) -> bool:
    def dfs(curr: Optional[TreeNode]) -> bool:
        def search(val: int) -> bool:
            tmp = root
            while tmp:
                if tmp.val == val and tmp != curr:
                    return True
                if tmp.val < val:
                    tmp = tmp.right
                else:
                    tmp = tmp.left
            return False

        if not curr:
            return False
        if search(k - curr.val):
            return True
        return dfs(curr.left) or dfs(curr.right)

    return dfs(root)


def gcdOfStrings(str1: str, str2: str) -> str:
    if str1 + str2 != str2 + str1:
        return ""
    gcd = math.gcd(len(str1), len(str2))
    return str1[:gcd]


def gcdOfStringsAlt(str1: str, str2: str) -> str:
    str1_len, str2_len = len(str1), len(str2)
    mn = min(str1_len, str2_len)

    def is_valid(gcd: int) -> bool:
        if str1_len % gcd or str2_len % gcd:
            return False
        f1, f2 = str1_len // gcd, str2_len // gcd
        base = str1[:gcd]
        return str1 == base * f1 and str2 == base * f2

    for i in range(mn, 0, -1):
        if is_valid(i):
            return str1[:i]
    return ""
