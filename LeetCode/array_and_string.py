import unittest
from operator import add
from typing import List

"""Introduction to Array"""


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


"""Introduction to 2D Array"""


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


"""Introduction to String"""


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


def str_str(haystack: str, needle: str) -> int:
    return haystack.find(needle)


def str_str_without_find(haystack: str, needle: str) -> int:
    index = 0
    needle_len = len(needle)
    haystack_len = len(haystack)
    if needle_len == 0:
        return 0
    for _ in haystack:
        if index + needle_len > haystack_len:
            return -1
        if haystack[index:index + needle_len] == needle:
            return index
        else:
            index += 1
    return -1


def str_str_without_find_better(haystack: str, needle: str) -> int:
    needle_len = len(needle)
    if needle_len == 0:
        return 0
    for index in range(len(haystack) - needle_len + 1):
        if haystack[index] != needle[0]:
            continue
        if haystack[index:index + needle_len] == needle:
            return index
    return -1


def longest_common_prefix(strings: List[str]) -> str:
    min_len_string = min(strings, key=len)
    prefix = []
    if len(strings) == 1:
        return strings[0]
    if any(element == "" for element in strings):
        return ''
    for i in range(len(min_len_string)):
        if not all(element[i] == min_len_string[i] for element in strings):
            return ''.join(prefix)
        prefix.append(min_len_string[i])


def longest_common_prefix_lex(strings: List[str]) -> str:
    min_lex, max_lex = min(strings), max(strings)
    for index, value in enumerate(min_lex):
        if max_lex[index] != value:
            return min_lex[:index]
    return min_lex


"""Two Pointer Technique"""


def reverse_string(s: List[str]) -> None:
    front_runner = 0
    back_runner = len(s) - 1
    while front_runner < back_runner:
        s[front_runner], s[back_runner] = s[back_runner], s[front_runner]
        front_runner += 1
        back_runner -= 1


def array_pair_sum(nums: List[int]) -> int:
    nums.sort()
    i = 0
    j = 1
    max_sum = 0
    while j < len(nums):
        max_sum += (nums[i], nums[j])
        i += 2
        j += 2
    return max_sum


def array_pair_sum_pythonic(nums: List[int]) -> int:
    return sum(sorted(nums)[::2])


def two_sum(numbers: List[int], target: int) -> List[int]:
    left = 0
    right = len(numbers) - 1
    while left < right:
        if numbers[left] + numbers[right] < target:
            left += 1
        elif numbers[left] + numbers[right] == target:
            return [left + 1, right + 1]
        else:
            right -= 1


def two_sum_dict(numbers: List[int], target: int) -> List[int]:
    dict_ = {}
    for index, value in enumerate(numbers):
        if target - value in dict_:
            return [dict_[target - value] + 1, index + 1]
        dict_[value] = index


def remove_element(nums: List[int], val: int) -> int:
    slow = 0
    for fast in nums:
        if fast != val:
            nums[slow] = fast
            slow += 1
    return slow


def remove_element_rare(nums: List[int], val: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        if nums[left] == val:
            nums[left], nums[right] = nums[right], nums[left]
            right -= 1
        else:
            left += 1
    return left


def find_max_consecutive_ones(nums: List[int]) -> int:
    max_ = 0
    sum_ = 0
    for fast in nums:
        if fast == 1:
            sum_ += 1
            # pythonic
            # max_ = max(sum_, max_)
            if sum_ > max_:
                max_ = sum_
        else:
            sum_ = 0
    return max_


def min_sub_array_len(target: int, nums: List[int]) -> int:
    res, total, i = len(nums) + 1, 0, 0
    for j in range(len(nums)):
        total += nums[j]
        while total >= target:
            res = min(res, j - i + 1)
            total -= nums[i]
            i += 1
    return 0 if res > len(nums) else res


def rotate(nums: List[int], k: int) -> None:
    nums_len = len(nums)
    k %= nums_len
    for _ in range(k):
        nums.insert(0, nums.pop())


def rotate_slicing(nums: List[int], k: int) -> None:
    k = k % len(nums)
    nums[:] = nums[-k:] + nums[:-k]


def get_row(row_index: int) -> List[int]:
    res = [1]
    for _ in range(row_index):
        res = list(map(add, [0] + res, res + [0]))
    return res


def get_row_lst_comp(row_index: int) -> List[int]:
    res = [1]
    for _ in range(row_index):
        res = [x + y for x, y in zip([0] + res, res + [0])]
    return res


def reverse_words(s: str) -> str:
    return ' '.join(s.split()[::-1])


def reverse_words_iii(s: str) -> str:
    return ' '.join([i[::-1] for i in s.split()])


def remove_duplicates(nums: List[int]) -> int:
    i = 1
    for value in nums[1:]:
        if nums[i - 1] != value:
            nums[i] = value
            i += 1
    return i


def move_zeros(nums: List[int]) -> None:
    i = 0
    for index, value in enumerate(nums):
        if value != 0:
            nums[i], nums[index] = nums[index], nums[i]
            i += 1


class Test(unittest.TestCase):

    def test_move_zero(self):
        nums = [0, 1, 0, 3, 12]
        move_zeros(nums)
        self.assertEqual([1, 3, 12, 0, 0], nums)
