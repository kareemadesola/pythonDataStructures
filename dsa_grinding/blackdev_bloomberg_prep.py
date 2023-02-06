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
