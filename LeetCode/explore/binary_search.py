import bisect
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


def find_peak_element(nums: List[int]) -> int:
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] > nums[mid + 1]:
            r = mid
        else:
            l = mid + 1
    return l


def findMin(nums: List[int]) -> int:
    # get no of rotations
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] > nums[r]:
            l = mid + 1
        else:
            r = mid
    return nums[l]


def searchRange(self, nums: List[int], target: int) -> List[int]:
    l, r = bisect.bisect_left(nums, target), bisect.bisect(nums, target)
    return [l, r - 1] if l != r else [-1, -1]


def findClosestElements(arr: List[int], k: int, x: int) -> List[int]:
    l, r = 0, len(arr) - k
    while l < r:
        mid = l + (r - l) // 2
        if x - arr[mid] > arr[mid + k] - x:
            l = mid + 1
        else:
            r = mid
    return arr[l : l + k]


def myPow(x: float, n: int) -> float:
    neg = n < 0
    n = abs(n)
    if n == 0:
        return 1
    res = 1
    while n != 0:
        if n % 2 == 0:
            x = x * x
            n //= 2
        else:
            res *= x
            n -= 1
    return 1 / res if neg else res


def isPerfectSquare(num: int) -> bool:
    l, r = 0, num
    while l < r:
        mid = l + (r - l) // 2
        if mid * mid < num:
            l = mid + 1
        else:
            r = mid

    return l * l == num


def nextGreatestLetter(letters: List[str], target: str) -> str:
    l, r = 0, len(letters)
    while l < r:
        mid = l + (r - l) // 2
        if letters[mid] <= target:
            l = mid + 1
        else:
            r = mid
    return letters[l % len(letters)]
