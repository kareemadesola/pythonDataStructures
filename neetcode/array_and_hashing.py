import heapq
from collections import Counter, defaultdict
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


def twoSum(nums: List[int], target: int) -> List[int]:
    seen = {}
    for idx, val in enumerate(nums):
        if target - val in seen:
            return [seen[target - val], idx]
        seen[val] = idx


def groupAnagrams(strs: List[str]) -> List[List[str]]:
    res = defaultdict(list)
    for val in strs:
        res[tuple(sorted(val))].append(val)
    return res.values()


def topKFrequent(nums: List[int], k: int) -> List[int]:
    cnt = Counter(nums)
    heap = [(-val, key) for key, val in cnt.items()]
    heapq.heapify(heap)

    res = []
    for _ in range(k):
        res.append(heapq.heappop(heap)[1])
    return res
