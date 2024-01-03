from typing import List


def removeDuplicates(nums: List[int]) -> int:
    l = 0
    for r in range(len(nums)):
        if nums[r] != nums[l]:
            l += 1
            nums[l] = nums[r]
    return l + 1


def removeElement(self, nums: List[int], val: int) -> int:
    l = 0
    for r_val in nums:
        if r_val != val:
            nums[l] = r_val
            l += 1
    return l


def strStr(haystack: str, needle: str) -> int:
    h_len, n_len = len(haystack), len(needle)
    for i in range(h_len - n_len + 1):
        if haystack[i:i + n_len] == needle:
            return i
    return -1


def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    while m > 0 and n > 0:
        if nums1[m - 1] > nums2[n - 1]:
            nums1[m + n - 1] = nums1[m - 1]
            m -= 1
        else:
            nums1[m + n - 1] = nums2[n - 1]
            n -= 1
    while n > 0:
        nums1[n - 1] = nums2[n - 1]
        n -= 1
