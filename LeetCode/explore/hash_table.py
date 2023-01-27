import collections
import unittest
from collections import Counter
from typing import List, Optional

from LeetCode.daily.july_22 import TreeNode

"""Practical Application Hash Set"""


# time O(k) where k is set list
# space O(k)
def contains_duplicate(nums: List[int]) -> bool:
    a = set()
    for i in nums:
        if i in a:
            return True
        a.add(i)


# time best, average and worst O(N) where N is list length
# space O(k)
def contains_duplicate_pythonic(nums: List[int]) -> bool:
    return not len(nums) != len(set(nums))


# time O(N) N is length of nums
# space O(k) k is the extra space - set
def single_number_set(nums: List[int]) -> int:
    return 2 * sum(set(nums)) - sum(nums)


# time O(N)
# space O(1)
def single_number_xor(nums: List[int]) -> int:
    res = 0
    for i in nums:
        res ^= i
    return res


# time O(max(nums1, nums2))
# space O(nums1)
def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
    return list(set(nums1).intersection(nums2))


# time 0(log n)
# space O(log n)
def is_happy_set(n: int) -> bool:
    def get_next(div) -> int:
        total_sum = 0
        while div != 0:
            div, mod = divmod(div, 10)
            total_sum += mod ** 2
        return total_sum

    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_next(n)

    return n == 1


# time O(log n)
# space O(1)
def is_happy_tortoise_hare(n: int) -> bool:
    def get_next(div: int) -> int:
        total_sum = 0
        while div != 0:
            div, mod = divmod(div, 10)
            total_sum += mod ** 2
        return total_sum

    slow = n
    fast = get_next(n)
    while n != 1 and slow != fast:
        slow = get_next(slow)
        fast = get_next(get_next(fast))
    return n == 1


# time O(log n)
# space O(1)
def is_happy_hardcoded_cycle(n: int) -> bool:
    cycle_members = {4, 16, 37, 58, 89, 145, 42, 20}

    def get_next(div: int) -> int:
        total_sum = 0
        while div != 0:
            div, mod = divmod(div, 10)
            total_sum += mod ** 2
        return total_sum

    while n != 1 and n not in cycle_members:
        n = get_next(n)

    return n == 1


# time O(log n)
# space O(1)
def is_happy_hardcoded_modified(n: int) -> bool:
    def get_next(div: int) -> int:
        total_sum = 0
        while div != 0:
            div, mod = divmod(div, 10)
            total_sum += mod ** 2
        return total_sum

    while n != 1 and n != 4:
        n = get_next(n)

    return n == 1


"""Practical application Hash map"""


# time O(nums)
# space O(nums)
def two_sum_two_pass(nums: List[int], target: int) -> List[int]:
    hash_map = {value: index for index, value in enumerate(nums)}
    for index, value in enumerate(nums):
        complement = target - value
        if complement in hash_map and hash_map[complement] != index:
            return [index, hash_map[complement]]


# time O(nums)
# space O(nums)
def two_sum_one_pass(nums: List[int], target: int) -> List[int]:
    hash_map = {}
    for index, value in enumerate(nums):
        complement = target - value
        if complement in hash_map:
            return [index, hash_map[complement]]
        hash_map[complement] = index


# time O(max(s,t))
# space O(s|t)
def is_isomorphic_two_dicts(s: str, t: str) -> bool:
    hash_map_one = {i: j for i, j in zip(s, t)}
    hash_map_two = {j: i for i, j in zip(s, t)}
    return ''.join(hash_map_one[i] for i in s) == t and ''.join(hash_map_two[j] for j in t) == s


def is_isomorphic_two_dicts_better(s: str, t: str) -> bool:
    hash_map_s_t = {}
    hash_map_t_s = {}

    for i, j in zip(s, t):
        if i not in hash_map_s_t and j not in hash_map_t_s:
            hash_map_s_t[i] = j
            hash_map_t_s[j] = i

        elif hash_map_s_t.get(i) != j or hash_map_t_s.get(j) != i:
            return False
    return True


def is_isomorphic_transform_string(s: str, t: str) -> bool:
    def transform_string(string) -> str:
        hash_map = {}
        new_str = []
        for index, value in enumerate(string):
            if value not in hash_map:
                hash_map[value] = index
            new_str.append(str(hash_map[value]))
        return " ".join(new_str)

    return transform_string(s) == transform_string(t)


# time O(max(len(list1),len(list2)))
# space O(min(len(list1),len(list2)))
def find_restaurant_set(list1: List[str], list2: List[str]) -> List[str]:
    common = set(list1).intersection(list2)
    hash_map = {value: index for index, value in enumerate(list1) if value in common}
    for index, value in enumerate(list2):
        if value in hash_map:
            hash_map[value] += index
    return [value for value, index in hash_map.items() if index == min(hash_map.values())]


# time O(max(list1,list2))
# space O(list1)
def find_restaurant(list1: List[str], list2: List[str]) -> List[str]:
    hash_map = {value: index for index, value in enumerate(list1)}
    min_, res = len(list1) + len(list2), []
    for index, value in enumerate(list2):
        i = hash_map.get(value, min_)
        if i + index < min_:
            min_ = i + index
            res = [value]
        elif i + index == min_:
            res.append(value)
    return res


# time O(s)
# space O(26) == O(1)
def first_uniq_char(s: str) -> int:
    hash_map = {}
    for i in s:
        if i not in hash_map:
            hash_map[i] = 1
        else:
            hash_map[i] += 1
    for index, value in enumerate(s):
        if hash_map[value] == 1:
            return index
    return -1


def first_uniq_char_counter(s: str) -> int:
    hash_map = Counter(s)
    for index, value in enumerate(s):
        if hash_map[value] == 1:
            return index
    return -1


# time O(max(nums1, nums2))
# space = O(nums1)
def intersect(nums1: List[int], nums2: List[int]) -> List[int]:
    hash_map = Counter(nums1)
    res = []
    for i in nums2:
        if i in hash_map and hash_map[i] > 0:
            res.append(i)
            hash_map[i] -= 1
    return res


def intersect_arrays_sorted(nums1: List[int], nums2: List[int]) -> List[int]:
    nums1_index, nums2_index = 0, 0
    res = []
    while nums1_index < len(nums1) and nums2_index < len(nums2):
        if nums1[nums1_index] < nums2[nums2_index]:
            nums1_index += 1
        elif nums1[nums1_index] == nums2[nums2_index]:
            res.append(nums1_index)
            nums1_index += 1
            nums2_index += 1
        else:
            nums2_index += 1

    return res


def contains_nearby_duplicate(nums: List[int], k: int) -> bool:
    dict_ = {}
    for index, value in enumerate(nums):
        if value in dict_ and index - dict_[value] <= k:
            return True
        dict_[value] = index
    return False


# time O(strs* KlogK) where K is the maximum length of the string in strs
# space O(strs * K)
def group_anagrams(strs: List[str]) -> List[List[str]]:
    hash_map = {}
    res: List[List[str]] = []
    for i in strs:
        sort_i = ''.join(sorted(i))
        if sort_i not in hash_map:
            hash_map[sort_i] = [i]
        else:
            hash_map[sort_i].append(i)
    for i in hash_map:
        res.append(hash_map[i])
    return res


def group_anagrams_default_dict(strs: List[str]) -> List[List[str]]:
    hash_map = collections.defaultdict(list)
    for i in strs:
        hash_map[tuple(sorted(i))].append(i)
    return hash_map.values()


# time O(strs * K)
def group_anagrams_count(strs: List) -> List[List[str]]:
    hash_map = collections.defaultdict(list)
    for i in strs:
        count = [0] * 26
        for c in i:
            count[ord(c) - ord('a')] += 1
        hash_map[tuple(count)].append(i)
    return hash_map.values()


# 2022-06-3, Fri, 9:29
# credit Neetcode
# time O(1)
# space O(1)
def is_valid_sudoku(board: List[List[str]]) -> bool:
    rows = collections.defaultdict(set)
    cols = collections.defaultdict(set)
    squares = collections.defaultdict(set)

    for r in range(9):
        for c in range(9):
            element = board[r][c]
            if element == '.':
                continue
            r_, c_ = r // 3, c // 3
            if element in rows[r] or element in cols[c] or element in squares[(r_, c_)]:
                return False
            rows[r].add(element)
            cols[c].add(element)
            squares[(r_, c_)].add(element)
    return True


def find_duplicate_subtrees(root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
    res: List[Optional[TreeNode]] = []
    cnt = collections.defaultdict(int)

    def traverse(node: TreeNode):
        if not node:
            return ''
        rep = f'({traverse(node.left)}){node.val}({traverse(node.right)})'
        cnt[rep] += 1
        if cnt[rep] == 2:
            res.append(node)
        return rep

    traverse(root)
    return res


class Test(unittest.TestCase):
    def test_find_restaurant(self):
        list1 = ["Shogun", "Tapioca Express", "Burger King", "KFC"]
        list2 = ["Piatti", "The Grill at Torrey Pines", "Hungry Hunter Steakhouse", "Shogun"]
        self.assertEqual(find_restaurant_set(list1, list2), ["Shogun"])

    def test_is_isomorphic(self):
        self.assertFalse(is_isomorphic_two_dicts('badc', 'baba'))
        self.assertTrue(is_isomorphic_two_dicts('egg', 'add'))

    def test_is_isomorphic_better(self):
        self.assertFalse(is_isomorphic_two_dicts_better('badc', 'baba'))
        self.assertTrue(is_isomorphic_two_dicts_better('egg', 'add'))

    def test_two_sum_add(self):
        nums = [2, 7, 11, 15]
        self.assertEqual(two_sum_two_pass(nums, 9), [0, 1] or [1, 0])
