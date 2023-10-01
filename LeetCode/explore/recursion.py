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
    memo = {0:0, 1:1}
    def dfs(val:int)->int:
        if val in memo:
            return memo[val]
        return dfs(val - 1) + dfs(val - 2)

    return dfs(n)
