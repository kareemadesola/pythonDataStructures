from typing import Optional


class ListNode:
    def __init__(self, x, next_node=None):
        self.val = x
        self.next: Optional[ListNode] = next_node


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


# time O(m+n) where m is len of l1 and n l2
# space O(1)
def get_intersection_node(head_a: ListNode, head_b: ListNode) -> Optional[ListNode]:
    l1, l2 = head_a, head_b
    while l1 != l2:
        l1 = l1.next if l1 else l2
        l2 = l2.next if l2 else l1
    return l1


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    l = r = head
    for _ in range(n):
        r = r.next
    # if r is None it means it is element
    # to delete is the first element
    if not r:
        head = head.next
        return head
    while r.next:
        l, r = l.next, r.next
    l.next = l.next.next
    return head


def remove_nth_from_end_dummy(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    # create dummy as to the point to prev
    dummy = ListNode(0, head)
    l, r = dummy, head
    for _ in range(n):
        r = r.next
    while r:
        l, r = l.next, r.next
    l.next = l.next.next
    return dummy.next
