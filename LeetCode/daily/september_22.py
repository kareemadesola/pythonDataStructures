import collections
from typing import List, Optional

from LeetCode.daily.july_22 import TreeNode


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
