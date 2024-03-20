import heapq
from collections import Counter, deque
from typing import List, Optional

from LeetCode.assessment.october_23_assessment import ListNode


def maxFrequencyElements(nums: List[int]) -> int:
    cnt = Counter(nums)
    max_count = max(cnt.values())
    freq = 0
    for val in cnt.values():
        if val == max_count:
            freq += 1
    return freq * max_count


def getCommon(nums1: List[int], nums2: List[int]) -> int:
    res = set(nums1).intersection(nums2)
    return min(res) if res else -1


def getCommonAlt(nums1: List[int], nums2: List[int]) -> int:
    if nums1[-1] < nums2[0] or nums2[-1] < nums1[0]:
        return -1
    nums1_ptr = nums2_ptr = 0
    min_ptr = min(len(nums1), len(nums2))
    while nums1_ptr != min_ptr or nums2_ptr != min_ptr:
        if nums1[nums1_ptr] < nums2[nums2_ptr]:
            nums1_ptr += 1
        elif nums1[nums1_ptr] == nums2[nums2_ptr]:
            return nums1[nums1_ptr]
        else:
            nums2_ptr += 1
    return -1


def customSortString(order: str, s: str) -> str:
    char_to_idx = {val: idx for idx, val in enumerate(order)}
    return "".join(sorted(s, key=lambda x: char_to_idx[x] if x in char_to_idx else 201))


def removeZeroSumSublists(head: Optional[ListNode]) -> Optional[ListNode]:
    dummy = start = ListNode(next=head)
    while start:
        prefix_sum = 0
        end = start.next
        while end:
            prefix_sum += end.val
            if prefix_sum == 0:
                start.next = end.next
            end = end.next
        start = start.next
    return dummy.next


def removeZeroSumSublistsAlt(head: Optional[ListNode]) -> Optional[ListNode]:
    dummy = curr = ListNode(next=head)
    prefix_sum = 0
    prefix_sum_to_node = {}

    while curr:
        prefix_sum += curr.val
        prefix_sum_to_node[prefix_sum] = curr
        curr = curr.next

    curr = dummy
    prefix_sum = 0
    while curr:
        prefix_sum += curr.val
        curr.next = prefix_sum_to_node[prefix_sum].next
        curr = curr.next
    return dummy.next


def pivotInteger(n: int) -> int:
    def sum_n(first: int, last: int) -> int:
        n = last - first + 1
        return (n * (first + last)) // 2

    l, r = 1, n
    while l < r:
        mid = l + (r - l) // 2
        l_sum = sum_n(1, mid)
        r_sum = sum_n(mid, n)

        if l_sum < r_sum:
            l = mid + 1
        else:
            r = mid
    return l if sum_n(1, l) == sum_n(l, n) else -1


def numSubarraysWithSum(nums: List[int], goal: int) -> int:
    def num_subarray_less_equal(target: int) -> int:
        if target < 0:
            return 0
        res = curr_sum = l = 0
        for r in range(len(nums)):
            curr_sum += nums[r]
            while curr_sum > target:
                curr_sum -= nums[l]
                l += 1
            res += r - l + 1
        return res

    return num_subarray_less_equal(goal) - num_subarray_less_equal(goal - 1)


def numSubarraysWithSumAlt(nums: List[int], goal: int) -> int:
    # prefix sum
    freq = {0: 1}
    res = total = 0
    for val in nums:
        total += val
        if total - goal in freq:
            res += freq[total - goal]
        freq[total] = freq.get(total, 0) + 1
    return res


def productExceptSelf(nums: List[int]) -> List[int]:
    n = len(nums)
    res = [1] * n

    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]

    suffix = 1
    for i in range(n - 1, -1, -1):
        res[i] *= suffix
        suffix *= nums[i]

    return res


def findMaxLength(nums: List[int]) -> int:
    diff_to_idx = {0: -1}
    res = ones = zeros = 0
    for r, val in enumerate(nums):
        if val == 0:
            zeros += 1
        else:
            ones += 1
        diff = ones - zeros
        if diff not in diff_to_idx:
            diff_to_idx[diff] = r
        else:
            res = max(res, r - diff_to_idx[diff])
    return res


def leastInterval(tasks: List[str], n: int) -> int:
    count = Counter(tasks)
    max_heap = [-cnt for cnt in count.values()]

    time, q = 0, deque()
    while q or max_heap:
        time += 1
        if max_heap:
            cnt = 1 + heapq.heappop(max_heap)
            if cnt:
                q.append((cnt, time + n))
        if q and q[0][1] == time:
            heapq.heappush(max_heap, q.popleft()[0])
    return time


def mergeInBetween(list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
    curr_l1 = dummy = ListNode(next=list1)
    for _ in range(a):
        curr_l1 = curr_l1.next

    remove = curr_l1.next
    curr_l1.next = list2

    for _ in range(b - a):
        remove = remove.next
    attach_end = remove.next
    remove.next = None
    while curr_l1.next:
        curr_l1 = curr_l1.next
    curr_l1.next = attach_end

    return dummy.next
