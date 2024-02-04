from collections import Counter
from typing import List


def containsDuplicate(nums: List[int]) -> bool:
    seen = set()
    for val in nums:
        if val in seen:
            return True
        seen.add(val)
    return False


def isAnagram(s: str, t: str) -> bool:
    s_cnt = Counter(s)
    for t_val in t:
        s_cnt[t_val] -= 1

    for val in s_cnt.values():
        if val: return False
    return True
