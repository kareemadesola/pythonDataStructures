from typing import List


# time O(k) where k is set list
# space O(k)
def contains_duplicate(nums: List[int]) -> bool:
    a = set()
    for i in nums:
        if i in a:
            return True
        a.add(i)


# time best, average and worst O(N) where N is list length
# space O(k)
def contains_duplicate_pythonic(nums: List[int]) -> bool:
    return not len(nums) != len(set(nums))
