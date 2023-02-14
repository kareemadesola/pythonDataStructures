from typing import List


def nextGreaterElement(nums1: List[int], nums2: List[int]) -> List[int]:
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


def nextGreaterElementBetter(nums1: List[int], nums2: List[int]) -> List[int]:
    val_to_idx = {val: idx for idx, val in enumerate(nums1)}
    res = [-1] * len(nums1)

    stack = []
    for i in range(len(nums2)):
        curr = nums2[i]
        while stack and curr > stack[-1]:
            val = stack.pop()
            idx = val_to_idx[val]
            res[idx] = curr
        if curr in val_to_idx:
            stack.append(curr)
    return res
