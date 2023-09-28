from typing import List, Optional

from LeetCode.explore.linked_list import ListNode


def reverseString(s: List[str]) -> None:
    """
    Do not return anything, modify s in-place instead.
    """

    def dfs(l, r):
        if l >= r:
            return
        s[l], s[r] = s[r], s[l]
        dfs(l + 1, r - 1)

    dfs(0, len(s) - 1)


def swapPairsIter(head: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    prev, curr = dummy, head
    while curr and curr.next:
        next_pair = curr.next.next
        second = curr.next

        second.next = curr
        curr.next = next_pair
        prev.next = second

        prev = curr
        curr = next_pair
    return dummy.next


def swapPairs(head: Optional[ListNode]) -> Optional[ListNode]:
    def dfs(curr: Optional[ListNode]) -> Optional[ListNode]:
        if not curr or not curr.next:
            return curr
        second = curr.next
        curr.next = dfs(curr.next.next)
        second.next = curr
        return second

    return dfs(head)


def reverseListIter(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head:
        return
    prev, curr = None, head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
