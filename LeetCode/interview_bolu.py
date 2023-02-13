from typing import List


def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    # Mon, 13 Feb 2023  22:05:38
    # time O(M*N) => M = len(nums1)
    # space O(1) => N = len(nums2)
    res = [-1] * len(nums1)
    for idx, val in enumerate(nums1):
        for j in range(nums2.index(val) + 1, len(nums2)):
            if nums2[j] > val:
                res[idx] = nums2[j]
                break
    return res
