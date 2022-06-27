import heapq
import math
# time and space 0(1)
import unittest
from typing import List, Optional, Dict, Set


def divide(dividend: int, divisor: int) -> int:
    quotient = math.trunc(dividend / divisor)
    check = 2 ** 31
    if -check < quotient < check - 1:
        return quotient
    return check - 1 if quotient > 0 else -check


def divide_from_32(dividend: int, divisor: int) -> int:
    if dividend == -2147483648 and divisor == -1:
        return 2147483647
    a, b, res = abs(dividend), abs(divisor), 0
    for x in reversed(range(32)):
        if a >= b << x:
            res += 1 << x
            a -= b << x
    return res if (dividend > 0) == (divisor > 0) else -res


def divide_from_zero(dividend: int, divisor: int) -> int:
    if dividend == -2147483648 and divisor == -1:
        return 2147483647
    a, b, res = abs(dividend), abs(divisor), 0
    while a >= b:
        x = 0
        while a >= (b << x):
            res += 1 << x
            a -= b << x
            x += 1
    return res if dividend ^ divisor > 0 else -res


def running_sum_enumerate(nums: List[int]) -> List[int]:
    sum_ = 0
    for index, value in enumerate(nums):
        nums[index] += sum_
        sum_ += value
    return nums


# time 0(n)
# space O(1)
def running_sum(nums: List[int]) -> List[int]:
    sum_ = 0
    for index in range(len(nums)):
        nums[index], sum_ = sum_ + nums[index], sum_ + nums[index]
    return nums


def running_sum_better(nums: List[int]) -> List[int]:
    for index in range(1, len(nums)):
        nums[index] += nums[index - 1]
    return nums


# 2022-06-2, Thu, 6:55
# time O(N * M) where N is length of matrix and
# M is the minimum length of column
# space O(max(N[i]))
def transpose(matrix: List[List[int]]) -> List[List[int]]:
    return [list(i) for i in zip(*matrix)]


def transpose_without_zip(matrix: List[List[int]]) -> List[List[int]]:
    row_len, col_len = len(matrix), len(matrix[0])
    res = [[0 for _ in range(row_len)] for _ in range(col_len)]
    for r in range(row_len):
        for c in range(col_len):
            res[c][r] = matrix[r][c]
    return res


# 2022-06-3, Fri, 15:58
class NumMatrix:
    # time O(rows * cols)
    # space O(rows * cols)
    def __init__(self, matrix: List[List[int]]):
        rows, cols = len(matrix), len(matrix[0])
        self.sum_matrix = [[0] * (cols + 1) for _ in range(rows + 1)]

        for r in range(rows):
            prefix = 0
            for c in range(cols):
                prefix += matrix[r][c]
                self.sum_matrix[r + 1][c + 1] = prefix + self.sum_matrix[r][c + 1]

    # # time limit exceeded
    # def sum_region(self, row1: int, col1: int, row2: int, col2: int) -> int:
    #     sum_ = 0
    #     for i in range(row1, row2 + 1):
    #         sum_ += sum(self.matrix[i][col1:col2 + 1])
    #     return sum_

    # time O(1)
    # space O(1)
    # sum_matrix has rows and cols increased by 1
    def sum_region(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.sum_matrix[row2 + 1][col2 + 1] - self.sum_matrix[row1][col2 + 1] \
               - self.sum_matrix[row1 + 1][col1] + self.sum_matrix[row1][col1]


# 2022-06-4, Sat, 12:34
# complexity under review
# time O(n*n)
# space O(n*n)
def solve_n_queens(n: int) -> List[List[str]]:
    cols = set()
    pos_diags = set()
    neg_diags = set()

    res = []
    board = [['.'] * n for _ in range(n)]

    def backtrack(r: int):
        if r == n:
            copy = [''.join(row) for row in board]
            res.append(copy)
            return
        for c in range(n):
            if c in cols or r + c in pos_diags or r - c in neg_diags:
                continue
            cols.add(c)
            pos_diags.add(r + c)
            neg_diags.add(r - c)
            board[r][c] = 'Q'

            backtrack(r + 1)

            cols.remove(c)
            pos_diags.remove(r + c)
            neg_diags.remove(r - c)
            board[r][c] = '.'

    backtrack(0)
    return res


def total_n_queens(n: int) -> int:
    cols = set()
    pos_diags = set()
    neg_diags = set()
    res = 0

    def backtrack(r: int):
        if r == n:
            nonlocal res
            res += 1
            return
        for c in range(n):
            if c in cols or r + c in pos_diags or r - c in neg_diags:
                continue
            cols.add(c)
            pos_diags.add(r + c)
            neg_diags.add(r - c)

            backtrack(r + 1)

            cols.remove(c)
            pos_diags.remove(r + c)
            neg_diags.remove(r + c)

    backtrack(0)
    return res


# 2022-06-6, Mon, 16:6
class ListNode:
    def __init__(self, x: int):
        self.val = x
        self.next = None


# time O(n+1)
# space O(1)
def get_intersection_node(head_a: ListNode, head_b: ListNode) -> Optional[ListNode]:
    l1, l2 = head_a, head_b
    while l1 != l2:
        l1 = l1.next if l1 else head_b
        l2 = l2.next if l2 else head_a
    return l1


# 2022-06-7, Tue, 7:57
def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    while n > 0 and m > 0:
        if nums2[n - 1] >= nums1[m - 1]:
            nums1[(m + n - 1)] = nums2[n - 1]
            n -= 1
        else:
            nums1[(m + n - 1)] = nums1[m - 1]
            m -= 1
    while n > 0:
        nums1[(m + n - 1)] = nums2[n - 1]
        n -= 1


# 2022-06-9, Thu, 5:19
def remove_palindrome_sub(s: str) -> int:
    if not s:
        return 0
    front, back = 0, len(s) - 1
    while not front > back:
        if s[front] != s[back]:
            return 2
        front += 1
        back -= 1
    return 1


# 2022-06-9, Thu, 6:30
def two_sum(numbers: List[int], target: int) -> List[int]:
    front, back = 0, len(numbers) - 1
    while front < back:
        two_sum_ = numbers[front] + numbers[back]
        if two_sum_ < target:
            front += 1
        elif two_sum_ == target:
            return [front + 1, back + 1]
        else:
            back -= 1


# 2022-06-10, Fri, 6:53
# time O(len_** 3)
# space O(1)
def length_of_longest_substring(s: str) -> int:
    len_ = len(s)
    if len_ <= 1:
        return len_
    l, r, res, temp = 0, 1, 0, 0
    while r < len_:
        if s[r] not in s[l:r]:
            r += 1
            temp += 1
            if temp > res:
                res = temp

        else:
            temp = 0
            l += 1
            r = l + 1
    return res + 1


def length_of_longest_substring_set(s: str) -> int:
    char_set = set()
    l, res = 0, 0
    for r in range(len(s)):
        while s[r] in char_set:
            char_set.remove(s[l])
            l += 1
        char_set.add(s[r])
        res = max(res, r - l + 1)
    return res


# time O(s)
# space O(min(hash_map, s)
def length_of_longest_substring_hash_map(s: str) -> int:
    hash_map = {}
    l, res = 0, 0
    for r, val in enumerate(s):
        if val in hash_map and l <= hash_map[val]:
            l = hash_map[val] + 1
        hash_map[val] = r
        res = max(res, r - l + 1)
    return res


def min_operations(nums, x):
    k = sum(nums) - x
    n = len(nums)
    i = prefix_sum = maximum = 0

    if k < 0:
        return -1
    if k == 0:
        return n
    for j in range(n):
        prefix_sum += nums[j]
        while prefix_sum > k:
            prefix_sum -= nums[i]
            i += 1
        if prefix_sum == k:
            maximum = max(maximum, j - i + 1)
    return n - maximum if maximum else -1


# 2022-06-12, Sun, 12:29
# time O(nums)
# space O(min(seen,nums))
def maximum_unique_subarray(nums: List[int]) -> int:
    seen = {}
    l = temp = res = 0
    for r, val in enumerate(nums):
        if val in seen:
            while l <= seen[val]:
                temp -= nums[l]
                l += 1
        seen[val] = r
        temp += val
        res = max(res, temp)
    return res


def maximum_unique_subarray_dp(nums: List[int]) -> int:
    seen = {}
    n = len(nums)
    sum_array = [0] * (n + 1)
    l = res = sum_ = 0
    for r, val in enumerate(nums):
        sum_ += val
        sum_array[r + 1] = sum_

    for r, val in enumerate(nums):
        if val in seen and l <= seen[val]:
            l = seen[val] + 1
        seen[val] = r
        res = max(res, sum_array[r + 1] - sum_array[l])
    return res


# 2022-06-13, Mon, 22:11
# time O(n*n)
# space O(n)
def minimum_total(triangle: List[List[int]]) -> int:
    dp = triangle[-1]
    for row in triangle[-2::-1]:
        for idx in range(len(row)):
            dp[idx] = row[idx] + min(dp[idx], dp[idx + 1])
    return dp[0]


# 2022-06-14, Tue, 17:37
# time O(word1*word2)
# space O(word1)
def min_distance(word1: str, word2: str) -> int:
    N, M = len(word1), len(word2)
    dp = [0] * (N + 1)

    for i in range(M + 1):
        temp = [0] * (N + 1)
        for j in range(N + 1):
            if i == 0 or j == 0:
                temp[j] = i + j
            if word1[j - 1] == word2[i - 1]:
                temp[j] = dp[j - 1]
            else:
                temp[j] = 1 + min(temp[j - 1], dp[j])
        dp = temp
    return dp[-1]


def longest_str_chain(words: List[str]) -> int:
    s = set(words)
    memo = {}

    def rec(word):
        if word not in s: return 0
        if word in memo: return memo[word]
        N = len(word)
        mx = 0
        for i in range(N):
            mx = max(mx, rec(word[:i] + word[i + 1:]) + 1)
        memo[word] = mx
        return mx

    for w in words:
        rec(w)
    return max(memo.values())


def longest_str_chain_no_recursion(words: List[str]) -> int:
    words.sort(key=len)
    res = 0
    memo: Dict[str, int] = {}
    for word in words:
        memo[word] = 1
        for i in range(len(word)):
            nxt = word[:i] + word[i + 1:]
            if nxt in memo:
                memo[word] = max(memo[word], memo[nxt] + 1)
        res = max(res, memo[word])
    return res


def longest_palindrome_brute_force(s: str) -> str:
    N = len(s)
    mx, res = 0, ''
    for i in range(N):
        for j in range(i + 1, N + 1):
            # if s[i:j] == ''.join(reversed(s[i:j])):
            # string= s[i:j] string == string[::-1]
            if s[i:j] == s[j - 1 - N:i - 1 - N:-1] and j - i > mx:
                mx, res = j - 1, s[i:j]
    return res


def longest_palindrome(s: str) -> str:
    res_l = res_r = 0

    def palindrome(l: int, r: int):
        nonlocal res_l, res_r
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if r - l > res_r - res_l:
                res_l, res_r = l, r
            l -= 1
            r += 1

    for i in range(len(s)):
        palindrome(i, i)
        palindrome(i, i + 1)
    return s[res_l:res_r + 1]


# 2022-06-17, Fri, 7:58
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left: TreeNode = left
        self.right: TreeNode = right


# time O(n)
# space O(n)
def min_camera_cover(root: Optional[TreeNode]) -> int:
    res = 0
    # Node with or covered by camera
    covered: Set[Optional[TreeNode]] = set()
    # skip the leaf nodes and start from one level above
    covered.add(None)

    def dfs(node: TreeNode, parent: TreeNode = None):
        nonlocal res
        if not node:
            return
        dfs(node.left, node)
        dfs(node.right, node)

        if node.left not in covered or node.right not in covered:
            res += 1
            covered.add(node)
            covered.add(parent)
            covered.add(node.left)
            covered.add(node.right)

    dfs(root)
    return res if root in covered else res + 1


def min_camera_cover_boolean(root: TreeNode) -> int:
    res = 0

    def dfs(node: TreeNode) -> tuple[bool, bool]:
        nonlocal res
        # camera, monitor
        if not node:
            return False, True
        c1, m1 = dfs(node.left)
        c2, m2 = dfs(node.right)

        camera, monitor = False, False
        if c1 or c2:
            monitor = True
        if not (m1 and m2):
            camera, monitor = True, True
            res += 1
        return camera, monitor

    _, m = dfs(root)
    return res if m else res + 1


# time O(n)
# space O(1)
def min_camera_cover_states(root: Optional[TreeNode]) -> int:
    res = 0

    #  2 --> Node with camera
    #  1 --> Node covered by camera
    # 0 --> Node not covered by camera
    def dfs(node: TreeNode) -> int:
        nonlocal res
        if not node: return 1
        left = dfs(node.left)
        right = dfs(node.right)

        # if left == 0 and right == 0
        if not (left and right):
            res += 1
            return 2
        if left == 2 or right == 2:
            return 1
        return 0

    return res if dfs(root) else res + 1


# 2022-06-18, Sat, 14:8
#  TLE
# class WordFilter:
#     def __init__(self, words: List[str]):
#         self.words = words
#
#     def f(self, prefix: str, suffix: str) -> int:
#         res = -1
#         for idx, val in enumerate(self.words):
#             if val[:len(prefix)] == prefix and val[-len(suffix):] == suffix:
#                 res = idx
#         return res

# 2022-06-20, Mon, 7:36
class WordFilter:
    # time O(words*k) where k is length of word
    # space O(words)
    def __init__(self, words: List[str]):
        self.words: Dict[str, int] = {f'{word}#{word}': idx for idx, word in enumerate(words)}

    # time O(spx*word*words)
    # space O()
    def f(self, prefix: str, suffix: str) -> int:
        spx = f'{suffix}#{prefix}'
        for word, index in reversed(self.words):
            if spx in word: return index
        return -1


def minimum_length_encoding(words: List[str]) -> int:
    hash_set = set(words)
    for word in words:
        for k in range(1, len(word)):
            if word[k:] in hash_set:
                hash_set.discard(word[k:])
    return sum(len(word) + 1 for word in hash_set)


# 2022-06-21, Tue, 21:10
def furthest_building(heights: List[int], bricks: int, ladders: int) -> int:
    heap = []
    for i in range(len(heights) - 1):
        d = heights[i + 1] - heights[i]
        # better to continue if less
        # as it won't move down the code unnecessarily
        if d > 0:
            heapq.heappush(heap, d)
        if len(heap) > ladders:
            bricks -= heapq.heappop(heap)
        if bricks < 0:
            return i
    return len(heights) - 1


# 2022-06-22, Wed, 6:2
# time O(nums*log(nums))
# space O(n)
def find_kth_largest(nums: List[int], k: int) -> int:
    return sorted(nums)[-k]


# time O(nums)
# space O(n)
def find_kth_largest_quick_select(nums: List[int], k: int) -> int:
    k = len(nums) - k

    def quick_select(l: int, r: int):
        l_p = l
        for i in range(l, r):
            if nums[i] <= nums[r]:
                nums[i], nums[l_p] = nums[l_p], nums[i]
                l_p += 1
        nums[l_p], nums[r] = nums[r], nums[l_p]
        if l_p > k: return quick_select(l, l_p - 1)
        if l_p < k: return quick_select(l_p + 1, r)
        return nums[l_p]

    return quick_select(0, len(nums) - 1)


# 2022-06-23, Thu, 14:44:31
# time O(courses*log(courses))
# space O(courses)
def schedule_course(courses: List[List[int]]) -> int:
    courses.sort(key=lambda x: (x[1]))
    heap = []
    max_time = 0
    for time, end_time in courses:
        heapq.heappush(heap, -time)
        max_time += time
        if max_time > end_time:
            max_time += heapq.heappop(heap)
    return len(heap)


def is_possible(target: List[int]) -> bool:
    total = sum(target)
    target = [-a for a in target]
    heapq.heapify(target)
    while True:
        a = -heapq.heappop(target)
        total -= a
        if a == 1 or total == 1: return True
        if a < total or total == 0 or a % total == 0:
            return False
        a %= total
        total += a
        heapq.heappush(target, -a)


# time O(products * log(products))
# space O(1)
def suggested_products(products: List[str], search_word: str) -> List[List[str]]:
    res = []
    l, r = 0, len(products) - 1
    products.sort()
    for i in range(len(search_word)):
        c = search_word[i]
        while l <= r and (len(products[l]) <= i or products[l][i] != c):
            l += 1
        while l <= r and (len(products[r]) <= i or products[r][i] != c):
            r -= 1

        res.append([])
        rem = r - l + 1
        for j in range(min(3, rem)):
            res[-1].append(products[l + j])
    return res


# 2022-06-25, Sat, 12:44:23
# time O(nums)
# space O(1)
def check_possibility(nums: List[int]) -> bool:
    changed = False
    for i in range(len(nums) - 1):
        if nums[i] <= nums[i + 1]:
            continue
        if changed:
            return False
        # decrease left
        # i == 0 to not make it out of bound
        if i == 0 or nums[i + 1] >= nums[i - 1]:
            nums[i] = nums[i + 1]
        # Increase right
        else:
            nums[i + 1] = nums[i]
        changed = True
    return True


def max_score_brute_force(card_points: List[int], k: int) -> int:
    res, ln = 0, len(card_points)
    for i in range(k + 1):
        temp = sum(card_points[:i]) + sum(card_points[ln - k + i:])
        res = max(res, temp)
    return res


def max_score(card_points: List[int], k: int) -> int:
    l, r = 0, len(card_points) - k
    total = sum(card_points[r:])
    res = total

    while r < len(card_points):
        total += card_points[l] - card_points[r]
        res = max(res, total)
        l += 1
        r += 1
    return res


def min_partitions(n: str) -> int:
    return int(max(n))


class Test(unittest.TestCase):
    def test_max_score_brute_force(self):
        self.assertEqual(536, max_score_brute_force([96, 90, 41, 82, 39, 74, 64, 50, 30], 8))

    def test_longest_palindrome_brute_force(self):
        self.assertEqual('aba', longest_palindrome_brute_force('babad'))

    def test_minimum_total(self):
        self.assertEqual(minimum_total([[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]), 11)

    def test_min_operations(self):
        self.assertEqual(6, min_operations(
            [6016, 5483, 541, 4325, 8149, 3515, 7865, 2209, 9623, 9763, 4052, 6540, 2123, 2074, 765,
             7520, 4941, 5290, 5868, 6150, 6006, 6077, 2856, 7826, 9119], 31841))
        self.assertEqual(5, min_operations([3, 2, 20, 1, 1, 3], 10))
        self.assertEqual(2, min_operations([1, 1, 4, 2, 3], 5))
        self.assertEqual(3, min_operations([1, 1, 1], 3))

    def test_length_of_longest_substring_hash_map(self):
        # self.assertEqual(3, length_of_longest_substring_hash_map('abcabcbb'))
        # self.assertEqual(3, length_of_longest_substring_hash_map('pwwkew'))
        self.assertEqual(2, length_of_longest_substring_hash_map("abba"))

    def test_length_of_longest_substring_set(self):
        self.assertEqual(3, length_of_longest_substring_set('abcabcbb'))
        self.assertEqual(3, length_of_longest_substring_set('pwwkew'))
        self.assertEqual(2, length_of_longest_substring_set("abba"))

    def test_length_of_longest_substring(self):
        self.assertEqual(3, length_of_longest_substring('abcabcbb'))
        self.assertEqual(3, length_of_longest_substring('pwwkew'))

    def test_total_n_queens(self):
        self.assertEqual(2, total_n_queens(4))

    def test_running_sum_better(self):
        self.assertEqual([1, 3, 6, 10], running_sum_better([1, 2, 3, 4]))

    def test_running_sum_enumerate(self):
        self.assertEqual([1, 3, 6, 10], running_sum_enumerate([1, 2, 3, 4]))

    def test_divide_from_zero(self):
        self.assertEqual(5, divide_from_zero(15, 3))

    def test_divide_from_32(self):
        self.assertEqual(5, divide_from_32(15, 3))
