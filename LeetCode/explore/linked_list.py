import collections
import copy
from typing import Optional, List


class ListNode:
    def __init__(self, x=0, next_node=None):
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


# 2022-07-11, Mon, 16:47:32
# time O(n)
# space O(1)
def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """ Use three pointers prev, current and a nxt pointer
    to keep track of curr next """
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev


# 2022-07-11, Mon, 19:20:13
# time O(n)
# space O(n)
def reverse_list_recurse_helper(head: Optional[ListNode]) -> Optional[ListNode]:
    """Use a recursive helper function that take
     curr and prev as parameters"""

    def reverse(curr, prev):
        if not curr:
            return prev
        nxt = curr.next
        curr.next = prev
        return reverse(nxt, curr)

    return reverse(head, None)


# 2022-07-11, Mon, 19:26:21
# time O(n)
# space O(n)
def reverse_list_recursion(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head:
        return None

    new_head = head
    if head.next:
        new_head = reverse_list_recursion(head.next)
        head.next.next = head
    head.next = None
    return new_head


def remove_elements(head: Optional[ListNode], val: int) -> Optional[ListNode]:
    """
    Create a dummy where its next is head
    while curr is not node:
        keep a reference for next node
        if curr val is val:
            prev.next = nxt
        else:
            prev = curr
        curr = nxt
    """
    dummy = ListNode(0, next_node=head)
    prev, curr = dummy, head
    while curr:
        nxt = curr.next
        if curr.val == val:
            prev.next = nxt
        else:
            prev = curr
        curr = nxt
    return dummy.next


"""Similar"""


# 2022-07-16, Sat, 18:06:57
# time O(n)
# space O(1)
def odd_even_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Have pointers odd, even_head and even
    Iterate till odd and even get to tail
    Link new of odd to even_head
    return head
    """
    if not head:
        return None
    odd, even_head, even = head, head.next, head.next
    while even and even.next:
        odd.next = odd.next.next
        odd = odd.next
        even.next = even.next.next
        even = even.next
    odd.next = even_head
    return head


def split_list_to_parts_att(head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
    res: List[Optional[ListNode]] = []
    head_len, ptr = 0, head
    while ptr:
        head_len += 1
        ptr = ptr.next
    ptr = head
    div, mod = divmod(head_len, k)
    while ptr:
        if mod > 0:
            temp = div + 1
            mod -= 1
        else:
            temp = div
        nxt = ptr
        for _ in range(temp):
            nxt = ptr.next
            ptr = nxt
        ptr.next = None
        res.append(head)
        head = nxt

    while head_len < k:
        res.append(None)
        head_len += 1
    return res


def split_list_to_part(head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
    """
    Get the length of head

    Get the integer division and remainder

    While iterating through k:
        Initialize head and head_pointer

    While iterating through width (integer division) + remainder:
        head_pointer.next = head_pointer = ListNode(cur.val)

        curr = curr.next

    Append head

    return ans
    """
    pass


# 2022-07-21, Thu, 04:04:17
# time O(n)
# space O(1)
def is_palindrome(head: Optional[ListNode]) -> bool:
    """
    Get the middle node

    reverse from the middle

    compare head with reversed middle
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    def reverse(list_node: ListNode) -> ListNode:
        prev, curr = None, list_node
        while curr:
            nxt = curr.next
            curr.next = prev
            prev, curr = curr, nxt
        return prev

    slow = reverse(slow)
    while slow:
        if head.val != slow.val:
            return False
        head, slow = head.next, slow.next
    return True


# 2022-07-23, Sat, 21:09:18
# time O(list1+list2)
# space O(1)
def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Create a dummy to represent prev node
    create a pointer `tail` for dummy
    while l1 or l2 is not None:
        if l1 < l2:
            append l1 to tail pointer
            go to next node
        else:
            append l2
            go to next node
        move tail pointer to next node
    if l1 is not None:
        tail.next = l1
    else if l2:
        tail.next = l2
    return dummy.next
    """
    tail = dummy = ListNode()
    while list1 and list2:
        if list1.val < list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next
    tail.next = list1 if list1 else list2
    return dummy.next


def add_two_numbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Curry a dummy node and its pointer
    while l1 and l2 is not None:
        get div, mod of both nodes and div
        create next of tail
        move all nodes to next node
    while l1 is not None:
        get div, mod of l1 and div
        create next of tail
        move l1 and tail to next node
    do the same for l2
    if div is not zero:
        assign div to tail.next
    get dummy.next

    Alternatively
        While l1 or l2 or div is not None:
        check always if l1 or l2 else val=0
        and node is None else next
    """
    dummy = tail = ListNode()
    div = 0
    while l1 and l2:
        div, mod = divmod(l1.val + l2.val + div, 10)
        tail.next = ListNode(mod)
        l1, l2, tail = l1.next, l2.next, tail.next
    while l1:
        div, mod = divmod(l1.val + div, 10)
        tail.next = ListNode(mod)
        l1, tail = l1.next, tail.next
    while l2:
        div, mod = divmod(l2.val + div, 10)
        tail.next = ListNode(mod)
        l2, tail = l2.next, tail.next
    if div:
        tail.next = ListNode(div)
    return dummy.next


# 2022-07-25, Mon, 14:52:43
# time O(number of nodes of head)
# space O(1)
class Node:
    def __init__(self, val=0, prev=None, next_node=None, child=None):
        self.val: int = val
        self.prev: Node = prev
        self.next: Node = next_node
        self.child: Node = child


def flatten(head: Optional[Node]) -> Optional[Node]:
    curr = head
    while curr:
        if not curr.child:
            curr = curr.next
            continue
        temp = curr.child
        # Get last child node
        while temp.next:
            temp = temp.next
        # point next of last child node to curr next
        temp.next = curr.next
        # point curr.next.prev to temp if curr.next exists
        if curr.next:
            curr.next.prev = temp
        curr.next = curr.child
        curr.child.prev = curr
        curr.child = None
    return head


# time O(number of nodes)
# space O(number of branches)
def flatten_stack(head: Optional[Node]) -> Optional[Node]:
    """
    Using a stack
    while curr is not None:
        if curr node has child:
            push its next node if it exists
            point curr next to its child
            point prev of curr next to itself
            del curr child
        else if curr next is None and stack is not empty:
            point curr next to element popped
            point prev of curr next to itself
        move curr to next node
    """
    curr, stack = head, []
    while curr:
        if curr.child:
            if curr.next:
                stack.append(curr.next)
            curr.next = curr.child
            curr.next.prev = curr
            curr.child = None
        elif not curr.next and stack:
            curr.next = stack.pop()
            curr.next.prev = curr
        curr = curr.next
    return head


class RandomNode:
    def __init__(self, x: int = 0, next_node: Node = None, random: Node = None
                 ):
        self.val = x
        self.next = next_node
        self.random = random


# 2022-07-27, Wed, 19:40:18
def copy_random_list_deepcopy(head: Optional[Node]) -> Optional[Node]:
    """
    Solution with deepcopy
    """
    return copy.deepcopy(head)


def copy_random_list(head: Optional[RandomNode]) -> Optional[RandomNode]:
    """
    Two passes

    create the dict that maps old to new node
    In the first pass, populate dict
    In the second pass, get mapped node
    and populate next and random pointers
    """
    hash_map = {None: None}
    curr = head
    while curr:
        hash_map[curr] = RandomNode(curr.val)
        curr = curr.next

    curr = head
    while curr:
        new = hash_map[curr]
        new.next = hash_map[curr.next]
        new.random = hash_map[curr.random]
        curr = curr.next
    return hash_map[head]


def rotate_right_deque(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Use a deque to store nodes
    since k can be large make k = k%len(deque)
    append left what has been popped k number of times
    """
    if not head or not head.next:
        return head
    deque_ = collections.deque()
    curr = head
    while curr:
        temp = curr.next
        # to make it more space  efficient
        curr.next = None
        deque_.append(curr)
        curr = temp
    k %= len(deque_)
    for _ in range(k):
        deque_.appendleft(deque_.pop())
    head = curr = deque_[0]
    while deque_:
        curr.next = deque_.popleft()
        curr = curr.next
    return head
