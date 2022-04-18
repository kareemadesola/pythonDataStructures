import sys
from typing import List


class Solution:

    def checkIfExist(self, arr: List[int]) -> bool:
        hash_table = {arr[i]: i for i in range(len(arr))}
        for i in range(len(arr)):
            if (2 * arr[i] in hash_table or (arr[i] % 2 == 0 and arr[i] // 2 in hash_table)) \
                    and i != hash_table[arr[i]]:
                return True
        return False

    def divideArray(self, nums: List[int]) -> bool:
        if len(nums) % 2 != 0:
            return False
        for element in nums:
            if not nums.count(element) % 2 == 0:
                return False
        return True

    def maximumSubsequenceCount(self, text: str, pattern: str) -> int:
        pass

    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        count = 0
        max_ones = []
        for i in nums:
            if i != 1:
                max_ones.append(count)
                count = 0
                continue
            count += 1
        max_ones.append(count)
        return max(max_ones)

    def maximum_subarray_one(self, nums: List[int]) -> int:
        """

        :param nums: list[int]
        :return: int
        eg [-2, 1, -3, 4, -1, 2, 1, -5, 4] is 6
        as [4,-1, 2, 1] is the maximum subarray
        in the array
        """
        loc_sum = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                #  The flaw in this algorithm is that it doesn't look at the
                # long term e.g [5,4,-1,7,8] is 23 but gives 15 since
                # it stop as soon as the current is less than previous
                if loc_sum < sum(nums[i:j]):
                    loc_sum = sum(nums[i:j])
        return loc_sum

    def maximum_subarray_two(self, nums: List[int]) -> int:
        """

        :param nums: list[int]
        :return: int
        eg [-2, 1, -3, 4, -1, 2, 1, -5, 4] is 6
        as [4,-1, 2, 1] is the maximum subarray
        in the array
        """
        global_sum = 0
        for i in range(len(nums)):
            local_sum = nums[i]
            for j in range(i + 1, len(nums)):
                if global_sum < local_sum:
                    global_sum = local_sum
                local_sum += nums[j]
            if global_sum < local_sum:
                global_sum = local_sum
        return global_sum

    def maximum_subarray_three(self, nums: List[int]) -> int:
        """

        :param nums: list[int]
        :return: int
        eg [-2, 1, -3, 4, -1, 2, 1, -5, 4] is 6
        as [4,-1, 2, 1] is the maximum subarray
        in the array
        """
        global_max = -sys.maxsize - 1
        local_max = 0
        for i in range(len(nums)):
            local_max = max(nums[i], nums[i] + local_max)

    def convert(self, s: str, numRows: int) -> str:
        """

        :param s:
        :param numRows:
        :return:
        """
        output = ""
        for i in range(numRows):
            for j in range(len(s)):
                if j % (2 * numRows - 2) == i or (2 * numRows - 2 - i) % (2 * numRows - 2) == j:
                    output += s[j]
        return output

    def merge_sorted_array(self, nums1: List[int], m: int, nums2: List[int], n: int):
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # m -= 1
        # n -= 1
        while m + n - 1 >= 0:
            if m == 0 or n - 1 >= 0 and nums1[m - 1] < nums2[n - 1]:
                nums1[m + n - 1] = nums2[n - 1]
                n -= 1
            else:
                nums1[m + n - 1] = nums1[m - 1]
                m -= 1
        return nums1

    def removeElement(self, nums: List[int], val: int) -> int:
        fast_pointer = 0
        slow_pointer = len(nums) - 1
        while fast_pointer < slow_pointer:
            if nums[slow_pointer] == val:
                slow_pointer -= 1
            if nums[fast_pointer] == val:
                nums[fast_pointer], nums[slow_pointer] = nums[slow_pointer], nums[fast_pointer]
                slow_pointer -= 1
            fast_pointer += 1
        return slow_pointer + 1


if __name__ == '__main__':
    test = Solution()
    # print(test.findMaxConsecutiveOnes([1, 1, 0, 1, 1, 1, ]))
    # print(test.maximum_subarray_one([5, 4, -1, 7, 8]))
    # print(test.maximum_subarray_two([-2, 1, -3, 4, -1, 2, 1, -5, 4, 7]))
    # print(test.convert("werekd", 3))
    # print(test.convert("paypalishiring", 3))
    # print(test.merge_sorted_array([1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3))
    # print(test.merge_sorted_array([1], 1, [], 0))
    # print(test.removeElement([3, 2, 2, 3], 3))
    print(test.checkIfExist([-2, 0, 10, -19, 4, 6, -8]))
