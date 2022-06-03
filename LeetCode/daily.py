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


class Test(unittest.TestCase):
    def test_running_sum_better(self):
        self.assertEqual([1, 3, 6, 10], running_sum_better([1, 2, 3, 4]))

    def test_running_sum_enumerate(self):
        self.assertEqual([1, 3, 6, 10], running_sum_enumerate([1, 2, 3, 4]))

    def test_divide_from_zero(self):
        self.assertEqual(5, divide_from_zero(15, 3))

    def test_divide_from_32(self):
        self.assertEqual(5, divide_from_32(15, 3))
