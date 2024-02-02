from typing import List


def divideArray(nums: List[int], k: int) -> List[List[int]]:
    nums.sort()
    res = []
    for i in range(0, len(nums), 3):
        tmp = [nums[i], nums[i + 1], nums[i + 2]]
        if nums[i + 2] - nums[i] > k:
            return []
        res.append(tmp)
    return res
