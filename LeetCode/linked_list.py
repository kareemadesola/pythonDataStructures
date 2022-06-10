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


# time O(n) where n since the inner while loop run
# only where one condition is met
# space O(1)
def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if fast == slow:
            # find where head and slow intersect
            while slow.next:
                if head == slow:
                    return slow
                slow, head = slow.next, head.next
