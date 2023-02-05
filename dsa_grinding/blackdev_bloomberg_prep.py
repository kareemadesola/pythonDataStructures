from typing import List


def twoSum(nums: List[int], target: int) -> List[int]:
    # Sun, 05 Feb 2023  12:54:21
    # time O(nums)
    # space O(nums)
    seen = {}
    for idx, val in enumerate(nums):
        if target - val in seen:
            return [idx, seen[target - val]]
        seen[val] = idx
