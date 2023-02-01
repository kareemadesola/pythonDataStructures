import collections
from functools import lru_cache
from typing import List, Optional, Tuple, DefaultDict

from LeetCode.daily.july_22 import TreeNode
from LeetCode.daily.june_22 import ListNode


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
        tmp = binary_search(self.data[key], timestamp)
        if tmp < 0: return ''
        return self.data[key][tmp][0]


def three_sum_closet(nums: List[int], target: int) -> int:
    nums.sort()
    res = nums[0] + nums[1] + nums[2]
    for i in range(len(nums) - 2):
        l, r = i + 1, len(nums) - 1
        while l < r:
            current_sum = nums[i] + nums[l] + nums[r]
            if current_sum == target: return current_sum
            if abs(target - current_sum) < abs(target - res):
                res = current_sum
            if current_sum < target:
                l += 1
            else:
                r -= 1
    return res


def test_three_sum_closet():
    three_sum_closet([-1, 2, 1, -4], 1)


def find_target(root: TreeNode, k: int) -> bool:
    seen = set()

    def dfs(curr: TreeNode, target: int) -> bool:
        if not curr: return False
        if target - curr.val in seen:
            return True
        seen.add(curr.val)
        return dfs(curr.left, target) or dfs(curr.right, target)

    return dfs(root, k)


def break_palindrome(palindrome: str) -> str:
    len_ = len(palindrome)
    for i in range(len_ // 2):
        if palindrome[i] != 'a':
            return palindrome[:i] + 'a' + palindrome[i + 1:]
    return palindrome[:-1] + 'b' if palindrome[:-1] else ''


def increasing_triplet(nums: List[int]) -> bool:
    i = j = 2 ** 31 - 1
    for num in nums:
        if num <= i:
            i = num
        elif num <= j:
            j = num
        else:
            return True
    return False


def largest_perimeter(nums: List[int]) -> int:
    """ to form a triangle the largest side must be less than sum of the other two sides"""
    nums.sort(reverse=True)
    for i in range(len(nums) - 2):
        if nums[i] < nums[i + 1] + nums[i + 2]:
            return nums[i] + nums[i + 1] + nums[i + 2]
    return 0


def delete_node(node: ListNode):
    next_node = node.next
    node.val = next_node.val
    node.next = node.next.next


def check_if_pangram(sentence: str) -> bool:
    return len(set(sentence)) == 26


def top_k_frequent(words: List[str], k: int) -> List[str]:
    return [i[0] for i in collections.Counter(sorted(words)).most_common(k)
            ]


def int_to_roman(num: int) -> str:
    M = ['', 'M', 'MM', 'MMM']
    C = ['', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM']
    X = ['', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC']
    I = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX']
    return M[num // 1000] + C[num % 1000 // 100] + X[num % 100 // 10] + I[num % 10]


def contains_nearby_duplicate(nums: List[int], k: int) -> bool:
    seen = {}
    for idx, num in enumerate(nums):
        if num in seen and abs(idx - seen[num]) <= k:
            return True
        seen[num] = idx
    return False


def findErrorNums(self, nums: List[int]) -> List[int]:
    seen = set()
    gotten = False
    for i in nums:
        if gotten:
            seen.add(i)
        elif i in seen:
            gotten = True
            result = [i]
            seen.add(i)
    for i in range(1, len(nums) + 1):
        if i not in seen:
            result.append(i)
            return result


def array_strings_are_equal(word1: List[str], word2: List[str]) -> bool:
    return ''.join(word1) == ''.join(word2)


def check_sub_array_sum_tle(nums: List[int], k: int) -> bool:
    # TLE
    arr_len = len(nums)
    if arr_len < 2: return False
    l, r = 0, 1
    p_sum = nums[l]
    while True:
        p_sum += nums[r]
        if p_sum % k == 0:
            return True
        r += 1
        if r == arr_len:
            l += 1
            if l > arr_len - 2:
                return False
            r = l + 1
            p_sum = nums[l]


def check_sub_array_sum(nums: List[int], k: int) -> bool:
    hash_map = {0: 0}
    s = 0
    for i in range(len(nums)):
        s += nums[i]
        if s % k not in hash_map:
            hash_map[s % k] = i + 1
        elif hash_map[s % k] < i:
            return True
    return False


def largest_overlap(img1: List[List[int]], img2: List[List[int]]) -> int:
    d = collections.defaultdict(int)
    n = len(img1)
    a = [(i, j) for i in range(n) for j in range(n) if img1[i][j]]
    b = [(i, j) for i in range(n) for j in range(n) if img2[i][j]]

    ans = 0
    for t1 in a:
        for t2 in b:
            # how many 1s exist in this sliding pattern
            t3 = (t2[0] - t1[0], t2[1] - t1[1])
            d[t3] += 1
            ans = max(ans, d[t3])
    return ans


def group_anagrams(strs: List[str]) -> List[List[str]]:
    hash_map = collections.defaultdict(list)
    for i in strs:
        hash_map[tuple(sorted(i))].append(i)
    return list(hash_map.values())


def earliest_full_bloom(plantTime: List[int], growTime: List[int]) -> int:
    res = 0
    for grow, plant in sorted(zip(growTime, plantTime)):
        res = max(res, grow) + plant
    return res


def shortest_path(grid: List[List[int]], k: int) -> int:
    if len(grid) == 1 and len(grid[0]) == 1:
        return 0
    q = collections.deque([(0, 0, 0, k)])
    m, n = len(grid), len(grid[0])
    visited = {}

    while q:
        x, y, path, obstacle = q.popleft()
        if x < 0 or x == m or y < 0 or y == n:
            continue
        if x == m - 1 and y == n - 1:
            return path
        if grid[x][y] == 1:
            if obstacle > 0:
                obstacle -= 1
            else:
                continue

        if (x, y) in visited and visited[(x, y)] >= obstacle:
            continue
        visited[(x, y)] = obstacle

        q.append((x + 1, y, path + 1, obstacle))
        q.append((x - 1, y, path + 1, obstacle))
        q.append((x, y - 1, path + 1, obstacle))
        q.append((x, y + 1, path + 1, obstacle))
    return -1


def is_toeplitz_matrix(matrix: List[List[int]]) -> bool:
    # time O(m * n) where m is the row size
    # space O(m * n) and n is the column size
    m, n = len(matrix), len(matrix[0])
    hash_map = collections.defaultdict(list)

    # fill the hash_map with elements of the same diagonal
    for i in range(m):
        for j in range(n):
            # two elements are on the same diagonal
            # if the difference of the row and column position
            # are equal
            hash_map[(i - j)].append(matrix[i][j])

    for key in hash_map:
        # check if there is any element not homogenous with the rest
        if any(i != hash_map[key][0] for i in hash_map[key]):
            return False
    return True


def is_toeplitz_matrix_better(matrix: List[List[int]]) -> bool:
    # time O(m * n) where m is the row size
    # space O(m + n) and n is the column size
    m, n = len(matrix), len(matrix[0])
    hash_map = {}
    for i in range(m):
        for j in range(n):
            if hash_map[i - j] not in hash_map:
                hash_map[i - j] = matrix[i][j]
            elif hash_map[i - j] != matrix[i][j]:
                return False
    return True


def is_toeplitz_matrix_better_optimal(matrix: List[List[int]]) -> bool:
    # time O(m * n) where m is the row size
    # space O(1) and n is the column size
    m, n = len(matrix), len(matrix[0])
    return all(i == 0 or j == 0 or matrix[i - 1][j - 1] == matrix[i][j] for i in range(m) for j in range(n))
