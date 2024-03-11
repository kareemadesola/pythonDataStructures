from collections import Counter
from typing import List


def maxFrequencyElements(nums: List[int]) -> int:
    cnt = Counter(nums)
    max_count = max(cnt.values())
    freq = 0
    for val in cnt.values():
        if val == max_count:
            freq += 1
    return freq * max_count


def getCommon(nums1: List[int], nums2: List[int]) -> int:
    res = set(nums1).intersection(nums2)
    return min(res) if res else -1


def getCommonAlt(nums1: List[int], nums2: List[int]) -> int:
    if nums1[-1] < nums2[0] or nums2[-1] < nums1[0]:
        return -1
    nums1_ptr = nums2_ptr = 0
    min_ptr = min(len(nums1), len(nums2))
    while nums1_ptr != min_ptr or nums2_ptr != min_ptr:
        if nums1[nums1_ptr] < nums2[nums2_ptr]:
            nums1_ptr += 1
        elif nums1[nums1_ptr] == nums2[nums2_ptr]:
            return nums1[nums1_ptr]
        else:
            nums2_ptr += 1
    return -1


def customSortString(order: str, s: str) -> str:
    char_to_idx = {val: idx for idx, val in enumerate(order)}
    return "".join(sorted(s, key=lambda x: char_to_idx[x] if x in char_to_idx else 201))
