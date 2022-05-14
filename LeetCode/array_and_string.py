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


def dominant_index(nums: List[int]) -> int:
    max_index = 0
    largest = 0
    second_largest = 0
    for index, value in enumerate(nums):
        if value > largest:
            second_largest = largest
            largest, max_index = value, index
        elif value > second_largest:
            second_largest = value
    if largest >= 2 * second_largest:
        return max_index
    return -1


def dominant_index_hint(nums: List[int]) -> int:
    max_index = 0
    largest = 0
    for index, value in enumerate(nums):
        if value > largest:
            largest, max_index = value, index
    for index, value in nums:
        if 2 * value > largest and index != max_index:
            return -1
    return max_index


def plus_one(digits: List[int]) -> List[int]:
    return list(map(int, str(int(''.join(map(str, digits))) + 1)))


def plus_one_base(digits: List[int]) -> List[int]:
    nums = 0
    for i in range(len(digits)):
        nums += digits[i] * 10 ** (len(digits) - 1 - i)
    return list(map(int, str(nums + 1)))


def find_diagonal_order(mat: List[List[int]]) -> List[int]:
    row_len, col_len = len(mat), len(mat[0])
    res, intermediate = [], []
    for i in range(row_len + col_len - 1):
        # always clear intermediate
        intermediate.clear()

        # find out the head of the diagonal
        row_index = 0 if i < col_len else i - col_len + 1
        col_index = i if i < col_len else col_len - 1

        # iterate down the slope
        while row_index < row_len and col_index > -1:
            intermediate.append(mat[row_index][col_index])
            row_index += 1
            col_index -= 1

        if i % 2 == 0:
            res.extend(intermediate[::-1])
        else:
            res.extend(intermediate)
    return res


class Test(unittest.TestCase):
    def test_pivot_index(self):
        nums = [-1, -1, -1, -1, -1, 0]
        self.assertEqual(pivot_index(nums), 2)
