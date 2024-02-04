from typing import List


def containsDuplicate(nums: List[int]) -> bool:
    seen = set()
    for val in nums:
        if val in seen:
            return True
        seen.add(val)
    return False
