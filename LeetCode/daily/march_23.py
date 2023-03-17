import math
import random
from collections import defaultdict
from typing import List, Optional

from LeetCode.daily.july_22 import TreeNode
from LeetCode.explore.linked_list import ListNode


def sortArray(nums: List[int]) -> List[int]:
    def merge_sort(arr: List[int]):

        if len(arr) <= 1:
            return
        mid = len(arr) // 2
        l = arr[0: mid]
        r = arr[mid:]
        merge_sort(l)
        merge_sort(r)
        arr1_len, arr2_len = len(l), len(r)
        i = j = 0
        while i < arr1_len and j < arr2_len:
            if l[i] < r[j]:
                arr[i + j] = l[i]
                i += 1
            else:
                arr[i + j] = r[j]
                j += 1
        while i < arr1_len:
            arr[i + j] = l[i]
            i += 1
        while j < arr2_len:
            arr[i + j] = r[j]
            j += 1

    merge_sort(nums)
    return nums


def compress(chars: List[str]) -> int:
    idx = idx_ans = 0
    while idx < len(chars):
        curr_char = chars[idx]
        cnt = 0
        while idx < len(chars) and chars[idx] == curr_char:
            idx += 1
            cnt += 1
        chars[idx_ans] = curr_char
        idx_ans += 1
        if cnt != 1:
            for char in str(cnt):
                chars[idx_ans] = char
                idx_ans += 1
    return idx_ans


def strStr(haystack: str, needle: str) -> int:
    return haystack.find(needle)


def findKthPositive(arr: List[int], k: int) -> int:
    arr = set(arr)
    res = 0
    while k:
        res += 1
        if res not in arr:
            k -= 1
    return res


def findKthPositiveBetter(A, k):
    l, r = 0, len(A)
    while l < r:
        m = (l + r) // 2
        if A[m] - 1 - m < k:
            l = m + 1
        else:
            r = m
    return l + k


def minimumTime(time: List[int], totalTrips: int) -> int:
    l, r = 1, max(time) * totalTrips

    def time_enough(given_time: int) -> bool:
        actual_trips = 0
        for t in time:
            actual_trips += given_time // t
        return actual_trips >= totalTrips

    while l < r:
        mid = l + (r - l) // 2
        if time_enough(mid):
            r = mid
        else:
            l = mid + 1
    return l


def minEatingSpeed(piles: List[int], h: int) -> int:
    l, r = 1, max(piles)

    def possible(k: int) -> bool:
        return sum(math.ceil(pile / k) for pile in piles) <= h  # slower
        # actual_h = 0
        # for pile in piles:
        #     actual_h += pile / k
        # return h < actual_h

    while l < r:
        mid = l + (r - l) // 2
        if possible(mid):
            r = mid
        else:
            l = mid
    return l


class Solution:

    def __init__(self, head: Optional[ListNode]):
        self.data = []
        while head:
            self.data.append(head.val)
            head = head.next

    def getRandom(self) -> int:
        return random.choice(self.data)


def detectCycle(head: Optional[ListNode]) -> Optional[ListNode]:
    tortoise = hare = head
    while tortoise and hare.next:
        tortoise = tortoise.next
        hare = hare.next.next
        if tortoise == hare:
            while head != tortoise:
                head = head.next
                tortoise = tortoise.next
            return head


def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    def merge(node1: ListNode, node2: ListNode):
        dummy = tail = ListNode()
        while node1 and node2:
            if node1.val < node2.val:
                tail.next = node1
                node1 = node1.next
            else:
                tail.next = node2
                node2 = node2.next
            tail = tail.next
        if node1: tail.next = node1
        if node2: tail.next = node2
        return dummy.next

    if not lists:
        return None

    while len(lists) > 1:
        merged_lists = []

        for i in range(0, len(lists), 2):
            l1 = lists[i - 1]
            l2 = lists[i + 1] if i + 1 < len(lists) else None
            merged_lists.append(merge(l1, l2))
        lists = merged_lists
    return lists[0]


def isSymmetric(root: Optional[TreeNode]) -> bool:
    def dfs(left: TreeNode, right: TreeNode) -> bool:
        if not left and not right:
            return True
        if not left or not right:
            return False
        return left.val == right.val and dfs(left.left, right.right) and dfs(left.right, right.left)

    return dfs(root.left, root.right)


def buildTree(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    def array_to_tree(l: int, r: int) -> Optional[TreeNode]:
        if l > r:
            return None
        root = TreeNode(postorder.pop())
        idx = in_val_to_idx[root.val]
        root.right = array_to_tree(idx + 1, r)
        root.left = array_to_tree(l, idx - 1)
        return root

    in_val_to_idx = {val: idx for idx, val in enumerate(inorder)}
    return array_to_tree(0, len(postorder) - 1)


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.end_of_word = False


class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for char in word:
            curr = curr.children[char]
        curr.end_of_word = True

    def search(self, word: str) -> bool:
        curr = self.root
        for char in word:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        return curr.end_of_word

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for char in prefix:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        return True
