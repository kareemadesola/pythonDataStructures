import collections
from functools import lru_cache
from typing import List, Optional


# 2022-07-1, Fri, 22:23:09
# time O(nlogn) where n is length of boxTypes due to merge sort
# space O(n) Due to merge sort
# https://leetcode.com/problems/maximum-units-on-a-truck/discuss/999125/JavaPython-3-Sort-by-the-units-then-apply-greedy-algorithm.
# For better implementation
def maximum_units(box_types: List[List[int]], truck_size: int) -> int:
    res = i = 0
    box_types.sort(key=lambda x: -x[1])
    while truck_size > 0 and i < len(box_types):
        mn = min(box_types[i][0], truck_size)
        res += mn * box_types[i][1]
        truck_size -= mn
        i += 1
    return res


# 2022-07-2, Sat, 22:02:12
# time O(max(MlogM,NlogN))
# space O(max(M,N))
# https://leetcode.com/problems/maximum-area-of-a-piece-of-cake-after-horizontal-and-vertical-cuts/discuss/661673
# /JavaPython-3-Find-the-max-width-and-height.
def max_area(h: int, w: int, horizontal_cuts: List[int], vertical_cuts: List[int]) -> int:
    horizontal_cuts.sort(), vertical_cuts.sort()
    horizontal_cuts = [0] + horizontal_cuts + [h]
    vertical_cuts = [0] + vertical_cuts + [w]
    max_h = max_w = 0
    for i in range(1, len(horizontal_cuts)):
        max_h = max(max_h, abs(horizontal_cuts[i] - horizontal_cuts[i - 1]))

    for i in range(1, len(vertical_cuts)):
        max_w = max(max_w, vertical_cuts[i] - vertical_cuts[i - 1])

    return (max_h * max_w) % (10 ** 9 + 7)


# time O(N) N = len(nums)
# space O(N)
def wiggle_max_length(nums: List[int]) -> int:
    dp = [nums[i - 1] - nums[i] for i in range(1, len(nums)) if nums[i - 1] - nums[i] != 0]
    if not dp: return 1
    cur = 2
    for i in range(1, len(dp)):
        if dp[i - 1] ^ dp[i] < 0:
            cur += 1
    return cur


# 2022-07-5, Tue, 03:36:52
# time O(N) N = len(nums)
# space O(1)
def wiggle_max_length_up_down(nums: List[int]) -> int:
    if not nums:
        return 0
    up = down = 1
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            up = down + 1
        elif nums[i] < nums[i - 1]:
            down = up + 1
    return max(up, down)


def candy(ratings: List[int]) -> int:
    n = len(ratings)
    res = [1] * n

    # increment if current > prev
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            res[i] = max(res[i - 1] + 1, res[i])

    # increment if  current > next
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            res[i] = max(res[i + 1] + 1, res[i])

    return sum(res)


# 2022-07-5, Tue, 22:19:33
# time O(nums)
# space O(nums)
def longest_consecutive(nums: List[int]) -> int:
    nums = set(nums)
    longest = 0
    for i in nums:
        if i - 1 not in nums:
            temp = 1
            while i + 1 in nums:
                temp += 1
            longest = max(longest, temp)
    return longest


# 2022-07-6, Wed, 13:43:38
# time O(nlogn) where n = len(nums)
# space O(n)
def longest_consecutive_sort(nums: List[int]) -> int:
    if not nums:
        return 0
    nums.sort()
    longest, temp = 0, 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1]:
            continue
        if nums[i] != nums[i - 1] + 1:
            longest = max(longest, temp)
            temp = 1
        else:
            temp += 1
    return max(longest, temp)


# 2022-07-6, Wed, 06:47:49
# time O(n)
# space O(1) since max_size = 128
@lru_cache
def fib(n: int) -> int:
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib(n - 1) + fib(n - 2)


def fib_iter(n: int) -> int:
    a, b = 1, 0
    for i in range(n):
        a, b = a + b, a
    return b


# 2022-07-7, Thu, 08:40:54
# time O(s1*s2)
# space O(s1*s2)
def is_interleave(s1: str, s2: str, s3: str) -> bool:
    s1_len, s2_len = len(s1), len(s2)
    if s1_len + s2_len != len(s3):
        return False
    dp = set()

    def dfs(i, j):
        if i == s1_len and j == s2_len:
            return True
        if (i, j) in dp:
            return False
        if i < s1_len and s1[i] == s3[i + j] and dfs(i + 1, j):
            return True
        if j < s2_len and s2[j] == s3[i + j] and dfs(i, j + 1):
            return True
        dp.add((i, j))
        return False

    return dfs(0, 0)


# 2022-07-7, Thu, 14:08:24
# time O(s1*s2)
# space O(s1*s2)
def is_interleave_dp(s1: str, s2: str, s3: str) -> bool:
    if len(s1) + len(s2) != len(s3):
        return False

    dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    dp[len(s1)][len(s2)] = True

    for i in range(len(s1), -1, -1):
        for j in range(len(s2), -1, -1):
            if i < len(s1) and s1[i] == s3[i + j] and dp[i + 1][j]:
                dp[i][j] = True
            if j < len(s2) and s2[j] == s3[i + j] and dp[i][j + 1]:
                dp[i][j] = True
    return dp[0][0]


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right


def right_side_view(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    right = right_side_view(root.right)
    left = right_side_view(root.left)
    return [root.val] + right + left[len(right):]


def right_side_view_bfs(root: Optional[TreeNode]) -> List[int]:
    res = []
    q = collections.deque([root])
    while q:
        right_side = None
        q_len = len(q)

        for _ in range(q_len):
            node = q.popleft()
            if node:
                right_side = node
                q.append(node.left)
                q.append(node.right)
        if right_side:
            res.append(right_side.val)
    return res


# https://leetcode.com/problems/binary-tree-right-side-view/discuss/56064/5-9-Lines-Python-48%2B-ms
def right_side_view_bfs_alt(root: Optional[TreeNode]) -> List[int]:
    """Remove None's from level using list comprehension"""
    res = []
    level = [root] if root else None
    while level:
        res.append(level[-1].val)
        level = [child for node in level for child in (node.left, node.right) if child]
    return res


def right_side_view_dfs(root: Optional[TreeNode]) -> List[int]:
    """Using dfs get all right first
    After right  has been exceeded check left"""
    res = []
    len_res = 0

    def collect(node: TreeNode, depth):
        nonlocal len_res
        if not node:
            return
        if depth == len_res:
            res.append(node)
            len_res += 1
        collect(node.right, depth + 1)
        collect(node.left, depth + 1)

    collect(root, 0)
    return res


def make_square(matchsticks: List[int]) -> bool:
    # todo
    return sum(matchsticks) % 4 == 0


def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    # todo
    pass


def build_tree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    if not preorder or not inorder:
        return None
    root = TreeNode(preorder[0])
    mid = inorder.index(preorder[0])
    root.left = build_tree(preorder[1:mid + 1], inorder[:mid])
    root.right = build_tree(preorder[mid + 1:], inorder[mid + 1:])
    return root
