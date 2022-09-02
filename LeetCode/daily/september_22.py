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
