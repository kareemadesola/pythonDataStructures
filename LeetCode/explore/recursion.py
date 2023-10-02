from typing import List, Optional

from LeetCode.explore.binary_tree import TreeNode
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
    prev, curr = None, head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev


def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
    def dfs(prev, curr):
        if not curr:
            return prev
        nxt = curr.next
        curr.next = prev
        return dfs(curr, nxt)

    return dfs(None, head)


def searchBST(root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    def dfs(curr: Optional[TreeNode]) -> Optional[TreeNode]:
        if not curr:
            return
        if curr and curr.val == val:
            return curr
        return dfs(curr.left) or dfs(curr.right)

    return dfs(root)


def getRowIter(rowIndex: int) -> List[int]:
    res = [1]
    for _ in range(rowIndex):
        tmp_len = len(res) - 1
        tmp = [0] * tmp_len
        for i in range(tmp_len):
            tmp[i] = res[i] + res[i + 1]
        res = [1] + tmp + [1]
    return res


def getRow(rowIndex: int) -> List[int]:
    def dfs(r, curr_row):
        if r == rowIndex:
            return curr_row
        new_row = [1]
        for i in range(1, len(curr_row)):
            new_row.append(curr_row[i - 1] + curr_row[i])
        new_row.append(1)
        return dfs(r + 1, new_row)

    return dfs(0, [1])


def fib(n: int) -> int:
    memo = {0: 0, 1: 1}

    def dfs(val: int) -> int:
        if val in memo:
            return memo[val]
        memo[val] = dfs(val - 1) + dfs(val - 2)
        return memo[val]

    return dfs(n)


def climbStairs(n: int) -> int:
    memo = {1: 1, 2: 2}

    def dfs(val: int) -> int:
        if val in memo:
            return memo[val]
        memo[val] = dfs(val - 1) + dfs(val - 2)
        return memo[val]

    return dfs(n)


def maxDepth(root: Optional[TreeNode]) -> int:
    mx = 0

    def dfs(curr: Optional[TreeNode], depth):
        nonlocal mx
        if not curr:
            mx = max(mx, depth)
        else:
            dfs(curr.left, depth + 1)
            dfs(curr.right, depth + 1)

    dfs(root, 0)
    return mx


def myPow(x: float, n: int) -> float:
    if n == 0:
        return 1

    def dfs(val: float, pw: int) -> float:
        if pw == 1:
            return val
        if pw % 2 == 0:
            return dfs(val * val, pw // 2)
        return val * dfs(val, pw - 1)

    return dfs(x, n) if n > 0 else 1 / dfs(x, -n)


def mergeTwoListsIter(
    list1: Optional[ListNode], list2: Optional[ListNode]
) -> Optional[ListNode]:
    dummy = curr = ListNode()
    while list1 and list2:
        if list1.val <= list2.val:
            curr.next = list1
            list1 = list1.next
        else:
            curr.next = list2
            list2 = list2.next
        curr = curr.next
    curr.next = list1 if list1 else list2
    return dummy.next


def mergeTwoLists(
    list1: Optional[ListNode], list2: Optional[ListNode]
) -> Optional[ListNode]:
    def merge(l1, l2) -> Optional[ListNode]:
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val <= l2.val:
            l1.next = merge(l1.next, l2)
            return l1
        l2.next = merge(l1, l2.next)
        return l2

    return merge(list1, list2)


def kthGrammar(n: int, k: int) -> int:
    if n == 1:
        return 0
    parent = kthGrammar(n - 1, (k + 1) // 2)
    if parent == 1:
        return 1 if k % 2 == 1 else 0
    return 0 if k % 2 == 1 else 1
