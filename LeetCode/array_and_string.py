import unittest
from operator import add
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


def find_diagonal_order_simulation(mat: List[List[int]]) -> List[int]:
    row_len, col_len = len(mat), len(mat[0])
    row_index, col_index = 0, 0
    res = []

    # true means going up
    # false means going down
    direction = True

    while row_index < row_len and col_index < col_len:
        res.append(mat[row_index][col_index])

        new_row_index = row_index + (-1 if direction else 1)
        new_col_index = col_index + (1 if direction else -1)

        if new_row_index < 0 or new_row_index == row_len or new_col_index < 0 or new_col_index == col_len:
            # if initial direction was up
            if direction:
                row_index += (col_index == col_len - 1)
                col_index += (col_index < col_len - 1)
            else:
                col_index += (row_index == row_len - 1)
                row_index += (row_index < row_len - 1)
            direction = not direction
        else:
            row_index = new_row_index
            col_index = new_col_index
    return res


def find_diagonal_order_sum(mat: List[List[int]]) -> List[int]:
    d = {}
    res = []
    for row in range(len(mat)):
        for col in range(len(mat[0])):
            if row + col not in d:
                d[row + col] = [mat[row][col]]
            else:
                d[row + col].append(mat[row][col])
    for key, value in d.items():
        if key % 2 == 0:
            res.extend(d[key][::-1])
        else:
            res.extend(d[key])
    return res


def spiral_order(matrix: List[List[int]]) -> List[int]:
    if not matrix:
        return []
    return list(matrix.pop(0)) + spiral_order([*zip(*matrix)][::-1])


def spiral_order_simulation(matrix: List[List[int]]) -> List[int]:
    top = 0
    bottom = len(matrix) - 1
    left = 0
    right = len(matrix[0]) - 1

    ans = []
    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            ans.append(matrix[top][col])
        top += 1

        for row in range(top, bottom + 1):
            ans.append(matrix[row][right])
        right -= 1

        for col in reversed(range(left, right + 1)):
            ans.append(matrix[bottom][col])
        bottom -= 1

        for row in reversed(range(top, bottom + 1)):
            ans.append(matrix[row][left])
        left += 1

    return ans[:len(matrix) * len(matrix[0])]


def generate(num_rows: int) -> List[List[int]]:
    res = [[1]]
    for _ in range(num_rows):
        res.append(list(map(add, res[-1] + [0], [0] + res[-1])))
    return res[:num_rows]


def add_binary(a: str, b: str) -> str:
    return bin(int(a, 2) + int(b, 2))[2:]


def add_binary_carry(a: str, b: str) -> str:
    carry = 0
    a = list(a)
    b = list(b)

    res = []

    while a or b or carry:
        if a:
            carry + int(a.pop())
        if b:
            carry += int(b.pop())

        res.append(str(carry % 2))
        carry = carry // 2
    return "".join(res)[::-1]


def add_binary_carry_no_extra_space(a: str, b: str) -> str:
    carry = 0
    a_end = len(a) - 1
    b_end = len(b) - 1
    res = []

    while carry or a_end >= 0 or b_end >= 0:
        if a_end >= 0:
            carry += int(a[a_end])
            a_end -= 1
        if b_end >= 0:
            carry += int(b[b_end])
            b_end -= 1
        res.append(str(carry % 2))
        carry //= 2
    return "".join(res)[::-1]


class Test(unittest.TestCase):
    def test_add_binary_carry(self):
        self.assertEqual(add_binary_carry_no_extra_space('1', '11'), '100')
