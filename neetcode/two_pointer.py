from typing import List


def isPalindrome(s: str) -> bool:
    res = []
    for char in s:
        if char.isalnum():
            res.append(char.lower())
    l, r = 0, len(res) - 1
    while l < r:
        if res[l] != res[r]:
            return False
        l += 1
        r -= 1
    return True


def twoSum(numbers: List[int], target: int) -> List[int]:
    def binary_search(t_get, f_idx):
        l, r = 0, n - 1
        while l <= r:
            mid = l + (r - l) // 2
            if mid != f_idx and numbers[mid] == t_get:
                return mid
            elif numbers[mid] < t_get:
                l = mid + 1
            else:
                r = mid - 1

    n = len(numbers)
    for i in range(n):
        res = binary_search(target - numbers[i], i)
        if res != -1:
            return [i + 1, res + 1]


def twoSumAlt(numbers: List[int], target: int) -> List[int]:
    l, r = 0, len(numbers) - 1
    while numbers[l] + numbers[r] != target:
        if numbers[l] + numbers[r] < target:
            l += 1
        else:
            r -= 1
    return [l + 1, r + 1]


def threeSum(nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()

    for idx, val in enumerate(nums):
        if val > 0:
            break

        if idx > 0 and val == nums[idx - 1]:
            continue
        l, r = idx + 1, len(nums) - 1
        while l < r:
            three_sum = val + nums[l] + nums[r]
            if three_sum < 0:
                l += 1
            elif three_sum > 0:
                r += 1
            else:
                res.append([val, nums[l], nums[r]])
                l += 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1
    return res


def fourSum(nums: List[int], target: int) -> List[List[int]]:
    nums.sort()
    res, quad = [], []

    def k_sum(start: int, target: int, k: int):
        if k != 2:
            for i in range(start, len(nums) - k + 1):
                if i > start and nums[i] == nums[i - 1]:
                    continue
                quad.append(nums[i])
                k_sum(i + 1, target - nums[i], k - 1)
                quad.pop()
            return
        l, r = start, len(nums) - 1
        while l < r:
            if nums[l] + nums[r] < target:
                l += 1
            elif nums[l] + nums[r] > target:
                r -= 1
            else:
                res.append(quad + [nums[l], nums[r]])
                l += 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1

    k_sum(0, target, 4)
    return res
