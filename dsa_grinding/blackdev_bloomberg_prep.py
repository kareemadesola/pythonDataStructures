import itertools
import math
from typing import List, Optional

from LeetCode.explore.linked_list import ListNode


def twoSum(nums: List[int], target: int) -> List[int]:
    # Sun, 05 Feb 2023  12:54:21
    # time O(nums)
    # space O(nums)
    seen = {}
    for idx, val in enumerate(nums):
        if target - val in seen:
            return [idx, seen[target - val]]
        seen[val] = idx


def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    # Mon, 06 Feb 2023  21:52:53
    # time O(max(l1,l2))
    # space O(max(l1,l2))
    dummy = curr = ListNode()
    carry = 0

    while l1 or l2 or carry:
        l1Val = l1.val if l1 else 0
        l2Val = l2.val if l2 else 0
        column_sum = l1Val + l2Val + carry
        carry = column_sum // 10
        curr.next = ListNode(column_sum % 10)
        curr = curr.next

        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next


def lengthOfLongestSubstring(s: str) -> int:
    res = l = 0
    char_to_idx = {}
    for r in range(len(s)):
        if s[r] in char_to_idx and l <= char_to_idx[s[r]]:
            l = char_to_idx[s[r]] + 1
        res = max(res, r - l + 1)
        char_to_idx[s[r]] = r
    return res


def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    # Mon, 06 Feb 2023  23:55:59
    # time O(N) N=> len(nums1 + nums2)
    # space O(N)
    l = l1 = l2 = 0
    nums1_len, nums2_len = len(nums1), len(nums2)
    nums_1_2 = [0] * (nums1_len + nums2_len)
    nums1_2_len = len(nums_1_2)

    while l1 < nums1_len and l2 < nums2_len and l < nums1_2_len:
        if nums1[l1] <= nums2[l2]:
            nums_1_2[l] = nums1[l1]
            l1 += 1
        else:
            nums_1_2[l] = nums2[l2]
            l2 += 1
        l += 1

    if l < nums1_2_len:
        while l1 < nums1_len:
            nums_1_2[l] = nums1[l1]
            l1 += 1
            l += 1
        while l2 < nums2_len:
            nums_1_2[l] = nums2[l2]
            l2 += 1
            l += 1

    return (nums_1_2[nums1_2_len // 2] + nums_1_2[-(nums1_2_len // 2) - 1]) / 2


def findMedianSortedArraysOptimal(nums1: List[int], nums2: List[int]) -> float:
    if len(nums2) < len(nums1):
        nums1, nums2 = nums2, nums1
    nums1_len = len(nums1)
    nums2_len = len(nums2)
    total = nums1_len + nums2_len
    half = total // 2

    l, r = 0, nums1_len - 1

    while True:
        i = l + (r - l) // 2
        j = half - i - 2

        l_nums1 = nums1[i] if i >= 0 else float('-inf')
        r_nums1 = nums1[i + 1] if i + 1 < nums1_len else float('inf')
        l_nums2 = nums2[j] if j >= 0 else float('-inf')
        r_nums2 = nums2[j + 1] if j + 1 < nums2_len else float('inf')

        if l_nums1 <= r_nums2 and l_nums2 <= r_nums1:
            if total % 2:
                return min(r_nums1, r_nums2)
            return (max(l_nums1, l_nums2) + min(r_nums1, r_nums2)) / 2
        elif l_nums1 > r_nums2:
            r = i - 1
        else:
            l = i + 1


def longestPalindrome(s: str) -> str:
    res_l = res_r = 0

    def palindrome(l, r):
        nonlocal res_r, res_l
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if r - l > res_r - res_l:
                res_l, res_r = l, r
            l -= 1
            r += 1

    for i in range(len(s)):
        palindrome(i, i)
        palindrome(i, i + 1)
    return s[res_l:res_r + 1]


def reverse(x: int) -> int:
    res = str(x)[::-1].strip('0-')
    if not res or not -2 ** 31 <= int(res) <= 2 ** 31 - 1:
        return 0
    return int(res) if x > 0 else -int(res)


def reverse_better(x: int) -> int:
    sign = 1 if x > 0 else -1
    res = sign * int(str(abs(x))[::-1])
    return res if -2 ** 31 <= res <= 2 ** 31 - 1 else 0


def reverse_math(x: int) -> int:
    mx, mn = 2 ** 31 - 1, -2 ** 31
    res = 0
    while x:
        tmp = x % 10
        pop = tmp if not tmp else tmp - 10 if x < 0 else tmp
        x = math.trunc(x / 10)
        if res > math.trunc(mx / 10) or (res == math.trunc(mx / 10) and pop > 7):
            return 0
        if res < math.trunc(mn / 10) or (res == math.trunc(mn / 10) and pop < -8):
            return 0
        res = res * 10 + pop
    return res


def three_sum(nums: List[int]) -> List[List[int]]:
    # TLE
    seen = set()
    res = []
    for i in itertools.combinations(nums, 3):
        if not sum(i) and tuple(sorted(i)) not in seen:
            res.append(list(i))
            seen.add(tuple(sorted(i)))
    return res


def three_sum_better(nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()

    for idx, val in enumerate(nums):
        if idx > 0 and val == nums[idx - 1]:
            continue
        l, r = idx + 1, len(nums) - 1
        while l < r:
            tmp = val + nums[l] + nums[r]
            if tmp > 0:
                r -= 1
            elif tmp < 0:
                l += 1
            else:
                res.append([val, nums[l], nums[r]])
                l += 1
                while nums[l] == nums[l - 1] and l < r:
                    l += 1
    return res


def nextGreaterElements(nums: List[int]) -> List[int]:
    res = [-1] * len(nums)
    for idx, val in enumerate(nums):
        for val_2 in nums[idx + 1:] + nums[:idx]:
            if val_2 > val:
                res[idx] = val_2
                break
    return res
