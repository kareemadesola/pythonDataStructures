import collections
from typing import Optional, List


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


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right


def right_side_view(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    right = right_side_view(root.right)
    left = right_side_view(root.left)
    return [root.val] + right + left[len(right):]


def right_side_view_bfs(root: Optional[TreeNode]) -> List[int]:
    res = []
    q = collections.deque([root])
    while q:
        right_side = None
        q_len = len(q)

        for _ in range(q_len):
            node = q.popleft()
            if node:
                right_side = node
                q.append(node.left)
                q.append(node.right)
        if right_side:
            res.append(right_side.val)
    return res


# https://leetcode.com/problems/binary-tree-right-side-view/discuss/56064/5-9-Lines-Python-48%2B-ms
def right_side_view_bfs_alt(root: Optional[TreeNode]) -> List[int]:
    """Remove None's from level using list comprehension"""
    res = []
    level = [root] if root else None
    while level:
        res.append(level[-1].val)
        level = [child for node in level for child in (node.left, node.right) if child]
    return res


def right_side_view_dfs(root: Optional[TreeNode]) -> List[int]:
    """Using dfs get all right first
    After right  has been exceeded check left"""
    res = []
    len_res = 0

    def collect(node: TreeNode, depth):
        nonlocal len_res
        if not node:
            return
        if depth == len_res:
            res.append(node)
            len_res += 1
        collect(node.right, depth + 1)
        collect(node.left, depth + 1)

    collect(root, 0)
    return res
