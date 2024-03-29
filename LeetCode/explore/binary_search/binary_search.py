import bisect
import random
from collections import Counter
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


def findMin(nums: List[int]) -> int:
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] > nums[r]:
            l = mid + 1
        else:
            r = mid
    return nums[l]


def findMinDuplicates(nums: List[int]) -> int:
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] > nums[r]:
            l = mid + 1
        elif nums[mid] < nums[r]:
            r = mid
        else:
            r -= 1
    return nums[l]


def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
    return list(set(nums1).intersection(set(nums2)))


def intersect(nums1: List[int], nums2: List[int]) -> List[int]:
    if len(nums2) > len(nums1):
        nums1, nums2 = nums2, nums1
    num_to_count = Counter(nums1)
    res = []
    for num in nums2:
        if num in num_to_count and num_to_count[num] != 0:
            res.append(num)
            num_to_count[num] -= 1
    return res


def twoSum(numbers: List[int], target: int) -> List[int]:
    def binary_search(left, to_find):
        l, r = left, len(numbers) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if numbers[mid] == to_find:
                return mid + 1
            elif numbers[mid] < to_find:
                l = mid + 1
            else:
                r = mid - 1
        return -1

    for i in range(len(numbers)):
        res = binary_search(i + 1, target - numbers[i])
        if res != -1:
            return [i + 1, res]


def findDuplicate(nums: List[int]) -> int:
    slow = fast = 0
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow2 = 0
    while True:
        slow = nums[slow]
        slow2 = nums[slow2]
        if slow == slow2:
            return slow


def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    total = len(nums1) + len(nums2)
    half = total // 2

    if len(nums2) < len(nums1):
        nums1, nums2 = nums2, nums1

    l, r = 0, len(nums1) - 1
    while True:
        mid1 = (l + r) // 2
        mid2 = half - mid1 - 2

        nums1_left = nums1[mid1] if mid1 >= 0 else float("-inf")
        nums1_right = nums1[mid1 + 1] if mid1 + 1 < len(nums1) else float("inf")
        nums2_left = nums2[mid2] if mid2 >= 0 else float("-inf")
        nums2_right = nums2[mid2 + 1] if mid2 + 1 < len(nums2) else float("inf")

        if nums1_left <= nums2_right and nums2_left <= nums1_right:
            if total % 2 == 0:
                return (max(nums1_left, nums2_left) + min(nums1_right, nums2_right)) / 2
            return min(nums1_right, nums2_right)
        elif nums1_left > nums2_right:
            r = mid1 - 1
        else:
            l = mid1 + 1


def smallestDistancePair(nums: List[int], k: int) -> int:
    n = len(nums)
    nums.sort()
    l, r = 0, nums[-1] - nums[0]

    def count_less_than_or_equal_mid(m: int) -> int:
        count = j = 0
        for i in range(n):
            while j < n and nums[j] - nums[i] <= m:
                j += 1
            count += j - i - 1
        return count

    while l < r:
        mid = l + (r - l) // 2
        cnt = count_less_than_or_equal_mid(mid)

        if cnt < k:
            l = mid + 1
        else:
            r = mid

    return l


def splitArray(nums: List[int], k: int) -> int:
    l, r = max(nums), sum(nums)

    def can_split(largest):
        sub_array = 0
        tmp = 0
        for num in nums:
            tmp += num
            if tmp > largest:
                tmp = num
                sub_array += 1
        return sub_array + 1 <= k

    while l < r:
        mid = l + (r - l) // 2
        if can_split(mid):
            r = mid
        else:
            l = mid + 1
    return l
