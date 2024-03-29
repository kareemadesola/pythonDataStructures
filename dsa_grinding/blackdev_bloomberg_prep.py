import collections
import itertools
import math
from typing import List, Optional

from LeetCode.daily.july_22 import TreeNode
from LeetCode.explore.linked_list.linked_list import ListNode


def twoSum(nums: List[int], target: int) -> List[int]:
    # Sun, 05 Feb 2023  12:54:21
    # time O(nums)
    # space O(nums)
    seen = {}
    for idx, val in enumerate(nums):
        if target - val in seen:
            return [idx, seen[target - val]]
        seen[val] = idx


def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
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


def lengthOfLongestSubstring(s: str) -> int:
    res = l = 0
    char_to_idx = {}
    for r in range(len(s)):
        if s[r] in char_to_idx and l <= char_to_idx[s[r]]:
            l = char_to_idx[s[r]] + 1
        res = max(res, r - l + 1)
        char_to_idx[s[r]] = r
    return res


def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    # Mon, 06 Feb 2023  23:55:59
    # time O(N) N=> len(nums1 + nums2)
    # space O(N)
    l = l1 = l2 = 0
    nums1_len, nums2_len = len(nums1), len(nums2)
    nums_1_2 = [0] * (nums1_len + nums2_len)
    nums1_2_len = len(nums_1_2)

    while l1 < nums1_len and l2 < nums2_len and l < nums1_2_len:
        if nums1[l1] <= nums2[l2]:
            nums_1_2[l] = nums1[l1]
            l1 += 1
        else:
            nums_1_2[l] = nums2[l2]
            l2 += 1
        l += 1

    if l < nums1_2_len:
        while l1 < nums1_len:
            nums_1_2[l] = nums1[l1]
            l1 += 1
            l += 1
        while l2 < nums2_len:
            nums_1_2[l] = nums2[l2]
            l2 += 1
            l += 1

    return (nums_1_2[nums1_2_len // 2] + nums_1_2[-(nums1_2_len // 2) - 1]) / 2


def findMedianSortedArraysOptimal(nums1: List[int], nums2: List[int]) -> float:
    if len(nums2) < len(nums1):
        nums1, nums2 = nums2, nums1
    nums1_len = len(nums1)
    nums2_len = len(nums2)
    total = nums1_len + nums2_len
    half = total // 2

    l, r = 0, nums1_len - 1

    while True:
        i = l + (r - l) // 2
        j = half - i - 2

        l_nums1 = nums1[i] if i >= 0 else float('-inf')
        r_nums1 = nums1[i + 1] if i + 1 < nums1_len else float('inf')
        l_nums2 = nums2[j] if j >= 0 else float('-inf')
        r_nums2 = nums2[j + 1] if j + 1 < nums2_len else float('inf')

        if l_nums1 <= r_nums2 and l_nums2 <= r_nums1:
            if total % 2:
                return min(r_nums1, r_nums2)
            return (max(l_nums1, l_nums2) + min(r_nums1, r_nums2)) / 2
        elif l_nums1 > r_nums2:
            r = i - 1
        else:
            l = i + 1


def longestPalindrome(s: str) -> str:
    res_l = res_r = 0

    def palindrome(l, r):
        nonlocal res_r, res_l
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if r - l > res_r - res_l:
                res_l, res_r = l, r
            l -= 1
            r += 1

    for i in range(len(s)):
        palindrome(i, i)
        palindrome(i, i + 1)
    return s[res_l:res_r + 1]


def reverse(x: int) -> int:
    res = str(x)[::-1].strip('0-')
    if not res or not -2 ** 31 <= int(res) <= 2 ** 31 - 1:
        return 0
    return int(res) if x > 0 else -int(res)


def reverse_better(x: int) -> int:
    sign = 1 if x > 0 else -1
    res = sign * int(str(abs(x))[::-1])
    return res if -2 ** 31 <= res <= 2 ** 31 - 1 else 0


def reverse_math(x: int) -> int:
    mx, mn = 2 ** 31 - 1, -2 ** 31
    res = 0
    while x:
        tmp = x % 10
        pop = tmp if not tmp else tmp - 10 if x < 0 else tmp
        x = math.trunc(x / 10)
        if res > math.trunc(mx / 10) or (res == math.trunc(mx / 10) and pop > 7):
            return 0
        if res < math.trunc(mn / 10) or (res == math.trunc(mn / 10) and pop < -8):
            return 0
        res = res * 10 + pop
    return res


def three_sum(nums: List[int]) -> List[List[int]]:
    # TLE
    seen = set()
    res = []
    for i in itertools.combinations(nums, 3):
        if not sum(i) and tuple(sorted(i)) not in seen:
            res.append(list(i))
            seen.add(tuple(sorted(i)))
    return res


def three_sum_better(nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()

    for idx, val in enumerate(nums):
        if idx > 0 and val == nums[idx - 1]:
            continue
        l, r = idx + 1, len(nums) - 1
        while l < r:
            tmp = val + nums[l] + nums[r]
            if tmp > 0:
                r -= 1
            elif tmp < 0:
                l += 1
            else:
                res.append([val, nums[l], nums[r]])
                l += 1
                while nums[l] == nums[l - 1] and l < r:
                    l += 1
    return res


def nextGreaterElements(nums: List[int]) -> List[int]:
    res = [-1] * len(nums)
    for idx, val in enumerate(nums):
        for val_2 in nums[idx + 1:] + nums[:idx]:
            if val_2 > val:
                res[idx] = val_2
                break
    return res


def nextGreaterElementsStack(nums: List[int]) -> List[int]:
    nums_len = len(nums)
    res = [-1] * nums_len
    stack = []
    for i in range(2 * nums_len):
        while stack and nums[i % nums_len] > nums[stack[-1]]:
            res[stack.pop()] = nums[i % nums_len]
        stack.append(i % nums_len)
    return res


def isValid(s: str) -> bool:
    close_to_open = {')': '(', ']': '[', '}': '{'}
    stack = []
    for bracket in s:
        if bracket in '([{':
            stack.append(bracket)
        else:
            if stack and stack[-1] == close_to_open[bracket]:
                stack.pop()
            else:
                return False
    return not stack


def isValidBetter(s: str) -> bool:
    open_to_close = {'(': ')', '[': ']', '{': '}'}
    stack = []
    for bracket in s:
        if bracket in open_to_close:
            stack.append(open_to_close[bracket])
        else:
            if not stack or stack.pop() != bracket:
                return False
    return not stack


def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = tail = ListNode()
    while list1 and list2:
        if list1.val <= list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next
    tail.next = list1 if list1 else list2
    return dummy.next


def search(nums: List[int], target: int) -> int:
    # get position of the lowest number
    nums_len = len(nums)
    l, r = 0, nums_len - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] > nums[r]:
            l = mid + 1
        else:
            r = mid
    rot = l
    l, r = 0, nums_len - 1
    while l <= r:
        mid = l + (r - l) // 2
        real_mid = (mid + rot) % nums_len
        if nums[real_mid] < target:
            l = mid + 1
        elif nums[real_mid] > target:
            r = mid - 1
        else:
            return real_mid
    return -1


def searchRange(nums: List[int], target: int) -> List[int]:
    if not nums: return [-1, -1]

    def left_range(arr: List[int], val) -> int:
        l, r = 0, len(arr)
        while l < r:
            mid = l + (r - l) // 2
            if arr[mid] < val:
                l = mid + 1
            else:
                r = mid
        return -1 if l == len(nums) and nums[l] != target else l

    def right_range(arr: List[int], val) -> int:
        l, r = 0, len(arr)
        while l < r:
            mid = l + (r - l) // 2
            if arr[mid] <= val:
                l = mid + 1
            else:
                r = mid
        return -1 if l == 0 and nums[l - 1] != target else l

    return [left_range(nums, target), right_range(nums, target)]


def trap(height: List[int]) -> int:
    l, r = 0, len(height) - 1
    mx_l, mx_r = height[l], height[r]
    res = 0
    while l < r:
        if mx_l < mx_r:
            l += 1
            mx_l = max(mx_l, height[l])
            res += mx_l - height[l]
        else:
            r -= 1
            mx_r = max(mx_r, height[r])
            res += mx_r - height[r]
    return res


def groupAnagrams(strs: List[str]) -> List[List[str]]:
    word_to_anagrams = collections.defaultdict(list)
    for word in strs:
        word_to_anagrams[tuple(sorted(word))].append(word)
    return list(word_to_anagrams.values())


def maxSubArray(nums: List[int]) -> int:
    res = nums[0]
    curr_sum = 0
    for num in nums:
        if curr_sum < 0:
            curr_sum = 0
        curr_sum += num
        res = max(res, curr_sum)
    return res


def spiralOrderRecursive(matrix: List[List[int]]) -> List[int]:
    if not matrix:
        return []
    return list(matrix.pop()) + spiralOrderRecursive([*zip(*matrix)][::-1])


def spiralOrder(matrix: List[List[int]]) -> List[int]:
    top = left = 0
    button, right = len(matrix) - 1, len(matrix[0]) - 1
    res = []
    while top <= button and left <= right:
        for col in range(left, right + 1):
            res.append(matrix[top][col])
        top += 1

        for row in range(top, button + 1):
            res.append(matrix[row][right])
        right -= 1

        for col in range(right, left - 1, -1):
            res.append(matrix[button][col])
        button -= 1

        for row in range(button, top - 1, -1):
            res.append(matrix[row][left])
        left += 1
    return res[:len(matrix) * len(matrix[0])]


def merge(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort()
    res = [intervals[0]]
    for start, end in intervals[1:]:
        last_end = res[-1][1]
        if start <= last_end:
            res[-1][1] = max(last_end, end)
        else:
            res.append([start, end])
    return res


def exist(board: List[List[str]], word: str) -> bool:
    word_len = len(word)

    def dfs(i: int, j: int, k: int):
        if k == word_len:
            return True
        if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == word[k]:
            val = board[i][j]
            board[i][j] = '#'
            res = dfs(i - 1, j, k + 1) or dfs(i + 1, j, k + 1) or dfs(i, j - 1, k + 1) or dfs(i, j + 1, k + 1)
            board[i][j] = val
            return res

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == word[0] and dfs(i, j, 0):
                return True


def mergeSortedArray(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    m -= 1
    n -= 1
    while m != -1 and n != -1:
        if nums1[m] < nums2[n]:
            nums1[m + n + 1] = nums2[n]
            n -= 1
        else:
            nums1[m + n + 1] = nums1[m]
            m -= 1
    while n != -1:
        nums1[m + n + 1] = nums2[n]
        n -= 1


def reverseBetween(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    # reach node at position 'left'
    left_prev, curr = dummy, head
    for _ in range(left - 1):
        left_prev, curr = curr, curr.next

    # Now curr = 'left', left_prev='node before left'
    # reverse from left to right
    prev = None
    for _ in range(right - left + 1):
        tmp_next = curr.next
        curr.next = prev
        prev, curr = curr, tmp_next

    # update pointers
    left_prev.next.next = curr
    left_prev.next = prev
    return dummy.next


def isValidBST(root: Optional[TreeNode]) -> bool:
    def is_valid(node: Optional[TreeNode], l: int, r: int) -> bool:
        if not node:
            return True
        if not l < node.val < r:
            return False
        return is_valid(node.left, l, node.val) and is_valid(node.right, node.val, r)

    return is_valid(root, -2 ** 31 - 1, 2 ** 31)


def buildTree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    def array_to_tree(l: int, r: int) -> Optional[TreeNode]:
        nonlocal pre_idx
        if l > r: return None

        root_val = preorder[pre_idx]
        root = TreeNode(root_val)

        pre_idx += 1

        root.left = array_to_tree(l, in_val_to_idx[root_val] - 1)
        root.right = array_to_tree(in_val_to_idx[root_val] + 1, r)

        return root

    pre_idx = 0
    in_val_to_idx = {val: idx for idx, val in enumerate(inorder)}
    return array_to_tree(0, len(preorder) - 1)


def sortedListToBST(head: Optional[ListNode]) -> Optional[TreeNode]:
    def to_bst(node: ListNode, tail: ListNode = None):
        slow = fast = node
        if node == tail: return None
        while fast != tail and fast.next != tail:
            fast = fast.next.next
            slow = slow.next
        root = TreeNode(slow.val)
        root.left = to_bst(node, slow)
        root.right = to_bst(slow.next, tail)
        return root

    return to_bst(head)


def pathSum(root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
    res = []

    def dfs(curr: TreeNode, curr_sum: int, path: List[int]):
        if not curr:
            return
        path.append(curr.val)
        curr_sum -= curr.val
        if not curr.left and not curr.right:
            if curr_sum == 0:
                res.append(path.copy())
        else:
            dfs(curr.left, curr_sum, path)
            dfs(curr.right, curr_sum, path)
        path.pop()

    dfs(root, targetSum, [])
    return res


def flatten(root: Optional[TreeNode]) -> None:
    def dfs(curr: TreeNode) -> Optional[TreeNode]:
        if not curr:
            return
        left_tail = dfs(curr.left)
        right_tail = dfs(curr.right)

        if curr.left:
            left_tail.right = curr.right
            curr.right = curr.left
            curr.left = None
        return right_tail or left_tail or curr

    dfs(root)
