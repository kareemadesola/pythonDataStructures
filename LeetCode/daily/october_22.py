import collections
from functools import lru_cache
from typing import List, Optional, Tuple, DefaultDict

from LeetCode.daily.july_22 import TreeNode


def num_decodings_memo(s: str) -> int:
    dp = {len(s): 1}

    def dfs(i: int):
        if i in dp:
            return dp[i]
        if s[i] == '0':
            return 0

        res = dfs(i + 1)
        if i + 1 < len(s) and (s[i] == '1' or s[i] == '2' and s[i + 1] in '0123456'):
            res += dfs(i + 2)
        dp[i] = res
        return res

    return dfs(0)


def num_decodings_dp(s: str) -> int:
    dp = {len(s): 1}
    for i in range(len(s) - 1, -1, -1):
        if s[i] == '0':
            dp[i] = 0
        else:
            dp[i] = dp[i + 1]
        if i + 1 < len(s) and (s[i] == '1' or s[i] == '2' and s[i + 1] in '0123456'):
            dp[i] += dp[i + 2]
    return dp[0]


def test_num_decoding():
    assert num_decodings_memo('1263') == 3


def num_rolls_to_target(n: int, k: int, target: int) -> int:
    @lru_cache(None)
    def dfs(curr_val: int, remains: int):
        if curr_val == 0:
            return 1 if not remains else 0
        return sum(dfs(curr_val - 1, remains - i) for i in range(1, k + 1)) % (10 ** 9 + 7)

    return dfs(n, target)


def num_rolls_to_target_memo(n: int, k: int, target: int) -> int:
    def dfs(curr_val: int, remains: int):
        if curr_val == 0:
            return 1 if not remains else 0
        if (curr_val, remains) in memo: return memo[(curr_val, remains)]
        memo[(curr_val, remains)] = sum(dfs(curr_val - 1, remains - i) for i in range(1, k + 1)) % (10 ** 9 + 7)
        return memo[(curr_val, remains)]

    memo = {}
    return dfs(n, target)


def min_cost(colors: str, needed_time: List[int]) -> int:
    res = max_cost = 0
    for i in range(len(colors)):
        if i > 0 and colors[i] != colors[i - 1]:
            max_cost = 0
        res += min(max_cost, needed_time[i])
        max_cost = max(max_cost, needed_time[i])
    return res


def has_path_sum(root: Optional[TreeNode], target_sum: int) -> bool:
    def dfs(curr_node: TreeNode, curr_sum: int):
        if not curr_node:
            return False
        if not curr_node.left and not curr_node.right and curr_node.val == curr_sum:
            return True
        curr_sum -= curr_node.val
        return dfs(curr_node.left, curr_sum) or dfs(curr_node.right, curr_sum)

    return dfs(root, target_sum)


def has_path_sum_bfs(root: Optional[TreeNode], target_sum: int) -> bool:
    q = collections.deque([(root, target_sum)])

    while q:
        curr, target_sum = q.popleft()
        if not curr: continue
        if not curr.left and not curr.right and curr.val == target_sum:
            return True
        new_target_sum = target_sum - curr.val
        q.append((curr.left, new_target_sum))
        q.append((curr.right, new_target_sum))
    return False


def add_one_row(root: Optional[TreeNode], val: int, depth: int) -> Optional[TreeNode]:
    if depth == 1:
        root = TreeNode(val, root)
        return root
    q, curr_depth, depth = collections.deque([root]), 1, depth - 1
    while q:
        for i in range(len(q)):
            curr = q.popleft()
            if not curr: continue
            if curr_depth == depth:
                curr.left = TreeNode(val, curr.left)
                curr.right = TreeNode(val, right=curr.right)
            q.append(curr.left)
            q.append(curr.right)
        curr_depth += 1
    return root


class TimeMap:

    def __init__(self):
        self.data: DefaultDict[str, List[Tuple[str, int]]] = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.data[key].append((value, timestamp))

    def get(self, key: str, timestamp: int) -> str:
        def binary_search(arr: List[Tuple[str, int]], timestamp_) -> int:
            l, r = 0, len(arr) - 1
            while l <= r:
                mid = l + (r - l) // 2
                if arr[l][1] < timestamp_:
                    l = mid + 1
                elif arr[l][1] > timestamp_:
                    r = mid - 1
                else:
                    return l
            return r

        if not self.data[key]: return ''
        return self.data[key][binary_search(self.data[key], timestamp)][0]
