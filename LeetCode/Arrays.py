import unittest
from typing import List


def duplicateZeros(arr: List[int]) -> list[int]:
    """
    Do not return anything, modify arr in-place instead.
    """
    for i in range(len(arr)):
        if arr[i] == 0:
            arr.insert(i, 0)
            arr.pop()
            i += 2
    return arr


def validMountainArray(arr: List[int]) -> bool:
    start_pointer = 0
    end_pointer = -1
    arr_len = len(arr)
    if arr_len < 3:
        return False
    while arr[start_pointer] < arr[start_pointer + 1]:
        start_pointer += 1
        if start_pointer + 1 >= arr_len:
            return False
    while arr[end_pointer] < arr[end_pointer - 1]:
        end_pointer -= 1
        if -(end_pointer - 1) > arr_len:
            return False
    if start_pointer + (-end_pointer) == arr_len:
        return True
    return False


def validMountainArrayBetter(arr: List[int]) -> bool:
    arr_len = len(arr)
    i = 0
    while i + 1 < arr_len and arr[i] < arr[i + 1]:
        i += 1

    # peak can be start or last
    if i == 0 or i == arr_len - 1:
        return False

    while i + 1 < arr_len and arr[i] > arr[i + 1]:
        i += 1

    return i == arr_len - 1


def replaceElements(arr: List[int]) -> List[int]:
    arr_len = len(arr)
    for i in range(arr_len - 1):
        arr[i] = max(arr[i + 1:])
    arr[arr_len - 1] = -1
    return arr


def replaceElementsBetter(arr: List[int]) -> List[int]:
    max_value = -1
    arr_len = len(arr)
    for i in reversed(range(arr_len)):
        arr[i], max_value = max_value, max(max_value, arr[i])
    return arr


def removeDuplicates(nums: List[int]) -> int:
    slow_runner = 0
    for fast_runner in nums[1:]:
        if nums[slow_runner] != fast_runner:
            slow_runner += 1
            nums[slow_runner] = fast_runner
    return slow_runner + 1


def moveZeroes(nums: List[int]) -> None:
    slow_runner = 0
    for fast_runner in range(1, len(nums)):
        if nums[slow_runner] == 0 and nums[fast_runner] != 0:
            nums[slow_runner], nums[fast_runner] = \
                nums[fast_runner], nums[slow_runner]
            slow_runner += 1
        if nums[slow_runner] != 0:
            slow_runner += 1


def sortArrayByParity(nums: List[int]) -> List[int]:
    slow_runner = 0
    for fast_runner in range(len(nums)):
        if nums[slow_runner] % 2 == 0:
            slow_runner += 1
        elif nums[slow_runner] % 2 != 0 and nums[fast_runner] % 2 == 0:
            nums[slow_runner], nums[fast_runner] = \
                nums[fast_runner], nums[slow_runner]
            slow_runner += 1
    return nums


def sortArrayByParityBetter(nums: List[int]) -> List[int]:
    i, j = 0, len(nums) - 1
    while i < j:
        if nums[i] % 2 == 1 and nums[j] % 2 == 0:
            nums[i], nums[j] = nums[j], nums[i]
        if nums[i] % 2 == 0:
            i += 1
        if nums[j] % 2 == 1:
            j -= 1
    return nums


def remove_elements_by_skipping(nums: List[int], val: int) -> int:
    slow_runner = 0
    for fast_runner in range(len(nums)):
        if nums[fast_runner] != val:
            nums[slow_runner] = nums[fast_runner]
            slow_runner += 1
    return slow_runner


def remove_elements_by_swapping(nums: List[int], val: int) -> int:
    front_runner = 0
    back_runner = len(nums) - 1
    while front_runner <= back_runner:
        if nums[front_runner] == val:
            nums[front_runner] = nums[back_runner]
            back_runner -= 1
        else:
            front_runner += 1
    return front_runner


def height_checker(heights: List[int]) -> int:
    expected_height = sorted(heights)
    answer = 0
    for runner in range(len(heights)):
        if heights[runner] != expected_height[runner]:
            answer += 1
    return answer


def third_max(nums: List[int]) -> int:
    nums = set(nums)
    if len(nums) < 3:
        return max(nums)
    for _ in range(3):
        answer = max(nums)
        nums.remove(answer)
    return answer


def find_disappeared_number(nums: List[int]) -> List[int]:
    nums = set(nums)
    answer = []
    for i in range(1, len(nums) + 1):
        if i not in nums:
            answer.append(i)
    return answer


def find_disappeared_numbers_intuitive(nums):
    for i in range(len(nums)):
        index = abs(nums[i]) - 1
        nums[index] = -abs(nums[index])
    return [i + 1 for i in range(len(nums)) if nums[i] > 0]


def sorted_squares(nums: List[int]) -> List[int]:
    return [i * i for i in sorted(nums, key=abs)]


def sorted_squares_pointer(nums: List[int]) -> List[int]:
    res = []
    front_runner = 0
    back_runner = len(nums) - 1
    while front_runner <= back_runner:
        back_value = nums[back_runner] * nums[back_runner]
        front_value = nums[front_runner] * nums[front_runner]
        if back_value > front_value:
            res.append(back_value)
            back_runner -= 1
        else:
            res.append(front_value)
            front_runner += 1
    return res[::-1]


class Test(unittest.TestCase):

    def test(self):
        nums = [4, 3, 2, 7, 8, 2, 3, 1]
        # print(nums)
        print(sorted_squares_pointer(nums))
        # print(nums)
