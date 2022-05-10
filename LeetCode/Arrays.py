import unittest
from typing import List


class Solution:
    def duplicateZeros(self, arr: List[int]) -> list[int]:
        """
        Do not return anything, modify arr in-place instead.
        """
        for i in range(len(arr)):
            if arr[i] == 0:
                arr.insert(i, 0)
                arr.pop()
                i += 2
        return arr

    def validMountainArray(self, arr: List[int]) -> bool:
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

    def validMountainArrayBetter(self, arr: List[int]) -> bool:
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

    def replaceElements(self, arr: List[int]) -> List[int]:
        arr_len = len(arr)
        for i in range(arr_len - 1):
            arr[i] = max(arr[i + 1:])
        arr[arr_len - 1] = -1
        return arr

    def replaceElementsBetter(self, arr: List[int]) -> List[int]:
        max_value = -1
        arr_len = len(arr)
        for i in reversed(range(arr_len)):
            arr[i], max_value = max_value, max(max_value, arr[i])
        return arr

    def removeDuplicates(self, nums: List[int]) -> int:
        slow_runner = 0
        for fast_runner in nums[1:]:
            if nums[slow_runner] != fast_runner:
                slow_runner += 1
                nums[slow_runner] = fast_runner
        return slow_runner + 1

    def moveZeroes(self, nums: List[int]) -> None:
        slow_runner = 0
        for fast_runner in range(1, len(nums)):
            if nums[slow_runner] == 0 and nums[fast_runner] != 0:
                nums[slow_runner], nums[fast_runner] = \
                    nums[fast_runner], nums[slow_runner]
                slow_runner += 1
            if nums[slow_runner] != 0:
                slow_runner += 1

    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        slow_runner = 0
        for fast_runner in range(len(nums)):
            if nums[slow_runner] % 2 == 0:
                slow_runner += 1
            elif nums[slow_runner] % 2 != 0 and nums[fast_runner] % 2 == 0:
                nums[slow_runner], nums[fast_runner] = \
                    nums[fast_runner], nums[slow_runner]
                slow_runner += 1
        return nums

    def sortArrayByParityBetter(self, nums: List[int]) -> List[int]:
        i, j = 0, len(nums) - 1
        while i < j:
            if nums[i] % 2 == 1 and nums[j] % 2 == 0:
                nums[i], nums[j] = nums[j], nums[i]
            if nums[i] % 2 == 0:
                i += 1
            if nums[j] % 2 == 1:
                j -= 1
        return nums


class Test(unittest.TestCase):
    test = Solution()

    def test_sort_array_by_parity_better(self):
        nums = [0, 1]
        print(nums)
        self.test.sortArrayByParityBetter(nums)
        print(nums)
