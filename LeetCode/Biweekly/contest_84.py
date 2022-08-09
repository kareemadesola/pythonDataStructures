from typing import List


def mergeSimilarItems(items1: List[List[int]], items2: List[List[int]]) -> List[List[int]]:
    hash_map = {i[0]: i[1] for i in items1}
    for i in items2:
        if i[0] in hash_map:
            hash_map[i[0]] += i[1]
        else:
            hash_map[i[0]] = i[1]
    return [[i, hash_map[i]] for i in sorted(hash_map)]


def countBadPairs(nums: List[int]) -> int:
    hash_map = {idx: val for idx, val in enumerate(nums)}

    def filter_fun(read):
        pass

    return len(filter(filter_fun, hash_map))
