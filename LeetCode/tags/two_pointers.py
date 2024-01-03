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
