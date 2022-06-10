from typing import Optional


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next: Optional[ListNode] = None


# time O(n) where n is length of head
# space O(1)
def has_cycle(head: Optional[ListNode]) -> bool:
    if not head:
        return False
    slow = head
    fast = head.next
    while slow != fast:
        if not (fast and fast.next):
            return False
        fast = fast.next.next
        slow = slow.next
    return True


def has_cycle_better(head: Optional[ListNode]) -> bool:
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
