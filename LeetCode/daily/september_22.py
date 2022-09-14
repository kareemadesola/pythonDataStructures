import collections
from typing import List, Optional

from LeetCode.daily.july_22 import TreeNode


# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


def good_nodes(root: TreeNode) -> int:
    # 2022-09-1, Thu, 21:20:44
    # time O(n) is the total number of nodes
    # space O(h) where h is height of tree
    def dfs(curr: TreeNode, max_: int):
        if not curr: return 0
        res = curr.val >= max_
        max_ = max(max_, curr.val)
        res += dfs(curr.left, max_) + dfs(curr.right, max_)
        return res

    return dfs(root, root.val)


def average_of_levels(root: TreeNode) -> List[float]:
    queue, res = collections.deque([root]), []

    while queue:
        len_q = len(queue)
        res.append(sum(i.val for i in queue) / len_q)
        for _ in range(len_q):
            dequeue = queue.popleft()
            if dequeue.left: queue.append(dequeue.left)
            if dequeue.right: queue.append(dequeue.right)
    return res


def prune_tree(root: TreeNode) -> Optional[TreeNode]:
    # 2022-09-7, Wed, 08:50:53

    # time O(n) n is number of node
    # space O(n) if the tree is skewed
    # like a linked list with root at head
    if not root: return
    root.left = prune_tree(root.left)
    root.right = prune_tree(root.right)
    if not root.left and not root.right and not root.val: return
    return root


if __name__ == '__main__':
    prune_tree(TreeNode(1, right=TreeNode(left=TreeNode(),
                                          right=TreeNode(1))))


def tree2str(root: TreeNode) -> str:
    if not root: return ''
    left = f'({tree2str(root.left)})' if root.left or root.right else ''
    right = f'({tree2str(root.right)})' if root.right else ''
    return f'{root.val}{left}{right}'


def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    res = []

    def dfs(curr: Optional[TreeNode]):
        if not curr: return
        dfs(curr.left)
        res.append(curr.val)
        dfs(curr.right)

    dfs(root)
    return res


def inorder_traversal_iterative(root: Optional[TreeNode]) -> List[int]:
    res, stack = [], []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        temp = stack.pop()
        res.append(temp.val)
        root = temp.right
    return res


def level_order(root: 'Node') -> List[List[int]]:
    if not root: return []
    res: List[List[int]] = []
    q = collections.deque([root])

    while q:
        q_len = len(q)
        level = []
        for _ in range(q_len):
            curr = q.popleft()
            level.append(curr.val)
            q.extend(curr.children)
        res.append(level)
    return res


def bag_of_tokens_score(tokens: List[int], power: int) -> int:
    tokens.sort()
    curr, l, r = 0, 0, len(tokens) - 1

    while l <= r:
        if power >= tokens[l]:
            power -= tokens[l]
            l += 1
            curr += 1
        elif curr and l != r:
            power += tokens[r]
            r -= 1
            curr -= 1
        else:
            break
    return curr


def valid_utf8(data: List[int]) -> bool:
    def byte_chr(s: str):
        res = 0
        for i in s:
            if i == '0':
                break
            res += 1
        return res

    data = [f"{i:08b}"[-8:] for i in data]
    i, j = 0, len(data)
    while i < j:
        tmp = byte_chr(data[i])
        if i + tmp > j or tmp == 1 or tmp > 4: return False
        for k in range(1, tmp):
            x = i + k
            if data[x][0] != '1' or data[x][1] != '0':
                return False
        i = i + tmp if tmp else i + 1
    return True


def valid_utf8_bit_mask(data: List[int]) -> bool:
    def byte_chr(s: int):
        mask, res = 1 << 7, 0
        while mask & s:
            mask >>= 1
            res += 1
        return res

    i, j = 0, len(data)
    while i < j:
        tmp = byte_chr(data[i])
        if i + tmp > j or tmp == 1 or tmp > 4: return False
        for k in range(1, tmp):
            x = i + k
            if not (data[x] & 1 << 7 and not data[x] & 1 << 6):
                return False
        i = i + tmp if tmp else i + 1
    return True


def pseudo_palindromic_path(root: Optional[TreeNode]) -> int:
    """A path is pseudo-palindromic if it contains at most
    one digit has an odd frequency (path is a power of two)"""

    def dfs(node: TreeNode, path: int):
        nonlocal count
        if not node:
            return
        path ^= 1 << node.val
        if not node.left and not node.right and not path & path - 1:
            count += 1

        dfs(node.left, path)
        dfs(node.right, path)

    count = 0
    dfs(root, 0)
    return count
