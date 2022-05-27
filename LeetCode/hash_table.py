import unittest
from typing import List

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


class Test(unittest.TestCase):
    def test_is_isomorphic(self):
        self.assertFalse(is_isomorphic_two_dicts('badc', 'baba'))
        self.assertTrue(is_isomorphic_two_dicts('egg', 'add'))

    def test_is_isomorphic_better(self):
        self.assertFalse(is_isomorphic_two_dicts_better('badc', 'baba'))
        self.assertTrue(is_isomorphic_two_dicts_better('egg', 'add'))

    def test_two_sum_add(self):
        nums = [2, 7, 11, 15]
        self.assertEqual(two_sum_two_pass(nums, 9), [0, 1] or [1, 0])
