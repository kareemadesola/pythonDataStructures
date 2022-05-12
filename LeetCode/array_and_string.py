import unittest
from typing import List


def pivot_index(nums: List[int]) -> int:
    total = sum(nums)
    left_sum = 0
    for index, value in enumerate(nums):
        if left_sum == total - left_sum - value:
            return index
        left_sum += value
    return -1


class Test(unittest.TestCase):
    def test_pivot_index(self):
        nums = [-1, -1, -1, -1, -1, 0]
        self.assertEqual(pivot_index(nums), 2)
