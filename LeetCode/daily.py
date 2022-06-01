import math

# time and space 0(1)
import unittest
from typing import List


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


class Test(unittest.TestCase):
    def test_running_sum_better(self):
        self.assertEqual([1, 3, 6, 10], running_sum_better([1, 2, 3, 4]))

    def test_running_sum_enumerate(self):
        self.assertEqual([1, 3, 6, 10], running_sum_enumerate([1, 2, 3, 4]))

    def test_divide_from_zero(self):
        self.assertEqual(5, divide_from_zero(15, 3))

    def test_divide_from_32(self):
        self.assertEqual(5, divide_from_32(15, 3))