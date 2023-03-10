import math
import random
from typing import List, Optional

from LeetCode.explore.linked_list import ListNode


def sortArray(nums: List[int]) -> List[int]:
    def merge_sort(arr: List[int]):

        if len(arr) <= 1:
            return
        mid = len(arr) // 2
        l = arr[0: mid]
        r = arr[mid:]
        merge_sort(l)
        merge_sort(r)
        arr1_len, arr2_len = len(l), len(r)
        i = j = 0
        while i < arr1_len and j < arr2_len:
            if l[i] < r[j]:
                arr[i + j] = l[i]
                i += 1
            else:
                arr[i + j] = r[j]
                j += 1
        while i < arr1_len:
            arr[i + j] = l[i]
            i += 1
        while j < arr2_len:
            arr[i + j] = r[j]
            j += 1

    merge_sort(nums)
    return nums


def compress(chars: List[str]) -> int:
    idx = idx_ans = 0
    while idx < len(chars):
        curr_char = chars[idx]
        cnt = 0
        while idx < len(chars) and chars[idx] == curr_char:
            idx += 1
            cnt += 1
        chars[idx_ans] = curr_char
        idx_ans += 1
        if cnt != 1:
            for char in str(cnt):
                chars[idx_ans] = char
                idx_ans += 1
    return idx_ans


def strStr(haystack: str, needle: str) -> int:
    return haystack.find(needle)


def findKthPositive(arr: List[int], k: int) -> int:
    arr = set(arr)
    res = 0
    while k:
        res += 1
        if res not in arr:
            k -= 1
    return res


def findKthPositiveBetter(A, k):
    l, r = 0, len(A)
    while l < r:
        m = (l + r) // 2
        if A[m] - 1 - m < k:
            l = m + 1
        else:
            r = m
    return l + k


def minimumTime(time: List[int], totalTrips: int) -> int:
    l, r = 1, max(time) * totalTrips

    def time_enough(given_time: int) -> bool:
        actual_trips = 0
        for t in time:
            actual_trips += given_time // t
        return actual_trips >= totalTrips

    while l < r:
        mid = l + (r - l) // 2
        if time_enough(mid):
            r = mid
        else:
            l = mid + 1
    return l


def minEatingSpeed(piles: List[int], h: int) -> int:
    l, r = 1, max(piles)

    def possible(k: int) -> bool:
        return sum(math.ceil(pile / k) for pile in piles) <= h  # slower
        # actual_h = 0
        # for pile in piles:
        #     actual_h += pile / k
        # return h < actual_h

    while l < r:
        mid = l + (r - l) // 2
        if possible(mid):
            r = mid
        else:
            l = mid
    return l


class Solution:

    def __init__(self, head: Optional[ListNode]):
        self.data = []
        while head:
            self.data.append(head.val)
            head = head.next

    def getRandom(self) -> int:
        return random.choice(self.data)


def detectCycle(head: Optional[ListNode]) -> Optional[ListNode]:
    tortoise = hare = head
    while tortoise and hare.next:
        tortoise = tortoise.next
        hare = hare.next.next
        if tortoise == hare:
            while head != tortoise:
                head = head.next
                tortoise = tortoise.next
            return head
