from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def evaluate_tree(root: Optional[TreeNode]) -> bool:
    if not root:
        return False

    if not root.left and not root.right:
        return bool(root.val)

    l_sum = evaluate_tree(root.left)
    r_sum = evaluate_tree(root.right)

    return l_sum or r_sum if root.val == 2 else l_sum and r_sum
