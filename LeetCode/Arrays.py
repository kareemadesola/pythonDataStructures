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


class Test(unittest.TestCase):
    def test_valid_mountain_array(self):
        pass
