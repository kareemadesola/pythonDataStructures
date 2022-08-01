import unittest
from typing import List


def minimum_operations(nums: List[int]) -> int:
    len_nums, count = len(nums), 0
    nums, idx = sorted(nums), 0
    while nums[-1]:
        while nums[idx] == 0:
            idx += 1
        x = nums[idx]
        for i in range(idx, len_nums):
            nums[i] = nums[i] - x
        count += 1
    return count


class Test(unittest.TestCase):
    def test_minimum_numbers(self):
        self.assertEqual(1, minimum_operations([3, 3, 0]))
