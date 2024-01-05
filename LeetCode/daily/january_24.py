from collections import defaultdict, Counter
from typing import List


def findContentChildren(g: List[int], s: List[int]) -> int:
    g.sort()
    s.sort()
    content_children = 0
    for cookie in s:
        if content_children < len(g) and g[content_children] <= cookie:
            content_children += 1
    return content_children


def findMatrix(nums: List[int]) -> List[List[int]]:
    cnt = Counter(nums)
    res = []
    keys_to_delete = []
    while cnt:
        tmp = []
        for key, val in cnt.items():
            if val == 0:
                keys_to_delete.append(key)
            else:
                tmp.append(key)
                cnt[key] -= 1
        for key in keys_to_delete:
            del cnt[key]
        keys_to_delete.clear()
        if tmp:
            res.append(tmp)
    return res


def findMatrixAlt(nums: List[int]) -> List[List[int]]:
    cnt = defaultdict(int)
    res = []
    for n in nums:
        row = cnt[n]
        if len(res) == row:
            res.append([])
        res[row].append(n)
        cnt[n] += 1
    return res
