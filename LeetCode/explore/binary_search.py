import random
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


def search_rotated(nums: List[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        # check right portion
        if nums[mid] < nums[l]:
            if nums[mid] > target or nums[r] < target:
                r = mid - 1
            else:
                l = mid + 1
        # check left portion
        else:
            if nums[mid] < target or nums[l] > target:
                l = mid + 1
            else:
                r = mid - 1
    return -1


def is_bad_version(version: int) -> bool:
    return random.choice([True, False])


def first_bad_version(n: int) -> int:
    l, r = 1, n
    while l <= r:
        mid = l + (r - l) // 2
        if is_bad_version(mid):
            if mid - 1 > 0 and not is_bad_version(mid - 1):
                return mid
            r = mid - 1
        else:
            l = mid + 1


def first_bad_version_l_bound(n: int) -> int:
    l, r = 1, n
    while l < r:
        mid = l + (r - l) // 2
        if is_bad_version(mid):
            r = mid
        else:
            l = mid + 1
    return l
