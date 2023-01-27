from typing import List


def search(nums: List[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1


def my_sqrt(x: int) -> int:
    l, r = 0, x
    while l < r:
        mid = l + (r - l) // 2
        if mid * mid <= x:
            l = mid + 1
        else:
            r = mid
    return l - 1 if x > 1 else x
