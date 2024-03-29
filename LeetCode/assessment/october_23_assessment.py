# Definition for a binary tree node.
import collections
import heapq
import math
from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next: Optional[ListNode] = next


def findTarget(root: Optional[TreeNode], k: int) -> bool:
    def convert_to_list() -> List[int]:
        res = []
        q = collections.deque([root])
        while q:
            q_len = len(q)
            for _ in range(q_len):
                curr = q.popleft()
                res.append(curr.val)
                if curr.left:
                    q.append(curr.left)
                if curr.right:
                    q.append(curr.right)
        return res

    def two_sum(arr: List[int]) -> bool:
        seen = set()
        for val in arr:
            if k - val in seen:
                return True
            seen.add(val)
        return False

    return two_sum(convert_to_list())


def findTargetDFS(root: Optional[TreeNode], k: int) -> bool:
    seen = set()

    def dfs(curr: TreeNode) -> bool:
        if not curr:
            return False
        if k - curr.val in seen:
            return True
        seen.add(curr.val)
        return dfs(curr.left) or dfs(curr.right)

    return dfs(root)


def findTargetBST(root: Optional[TreeNode], k: int) -> bool:
    def dfs(curr: Optional[TreeNode]) -> bool:
        def search(val: int) -> bool:
            tmp = root
            while tmp:
                if tmp.val == val and tmp != curr:
                    return True
                if tmp.val < val:
                    tmp = tmp.right
                else:
                    tmp = tmp.left
            return False

        if not curr:
            return False
        if search(k - curr.val):
            return True
        return dfs(curr.left) or dfs(curr.right)

    return dfs(root)


def gcdOfStrings(str1: str, str2: str) -> str:
    if str1 + str2 != str2 + str1:
        return ""
    gcd = math.gcd(len(str1), len(str2))
    return str1[:gcd]


def gcdOfStringsAlt(str1: str, str2: str) -> str:
    str1_len, str2_len = len(str1), len(str2)
    mn = min(str1_len, str2_len)

    def is_valid(gcd: int) -> bool:
        if str1_len % gcd or str2_len % gcd:
            return False
        f1, f2 = str1_len // gcd, str2_len // gcd
        base = str1[:gcd]
        return str1 == base * f1 and str2 == base * f2

    for i in range(mn, 0, -1):
        if is_valid(i):
            return str1[:i]
    return ""


def romanToInt(s: str) -> int:
    roman_to_int = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}
    n = len(s)
    res = 0

    for i in range(n - 1):
        if not roman_to_int[s[i]] < roman_to_int[s[i + 1]]:
            res += roman_to_int[s[i]]
        else:
            res -= roman_to_int[s[i]]
    return res + roman_to_int[s[-1]]


def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    curr = dummy = ListNode()
    carry = 0

    while l1 or l2 or carry:
        l1_val = l1.val if l1 else 0
        l2_val = l2.val if l2 else 0
        carry, half_sum = divmod((l1_val + l2_val + carry), 10)
        curr.next = ListNode(half_sum)
        curr = curr.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next


def numUniqueEmails(emails: List[str]) -> int:
    seen = set()

    for email in emails:
        tmp = []
        idx_a = email.index("@")
        for char in email[:idx_a]:
            if char == "+":
                break
            if char == ".":
                continue
            tmp.append(char)
        seen.add("".join(tmp) + email[idx_a:])
    return len(seen)


def numUniqueEmailsAlt(emails: List[str]) -> int:
    seen = set()
    for mail in emails:
        local, domain = mail.split("@")
        local: str = local.split("+")[0].replace(".", "")
        seen.add((local, domain))
    return len(seen)


def totalFruit(fruits: List[int]) -> int:
    l = res = total = 0
    count = collections.defaultdict(int)

    for r in range(len(fruits)):
        count[fruits[r]] += 1
        total += 1

        while len(count) > 2:
            count[fruits[l]] -= 1
            if not count[fruits[l]]:
                del count[fruits[l]]
            l += 1
            total -= 1
        res = max(res, total)
    return res


def wordPattern(pattern: str, s: str) -> bool:
    s = s.split(" ")
    if len(pattern) != len(s):
        return False
    char_to_word = {}
    word_to_char = {}

    for char, word in zip(pattern, s):
        if char in char_to_word and char_to_word[char] != word:
            return False
        if word in word_to_char and word_to_char[word] != char:
            return False
        char_to_word[char] = word
        word_to_char[word] = char
    return True


def updateMatrix(mat: List[List[int]]) -> List[List[int]]:
    m, n = len(mat), len(mat[0])
    q = collections.deque([])
    DIR = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for row in range(m):
        for col in range(n):
            # marked as unsolved
            if mat[row][col]:
                mat[row][col] = -1
            else:
                q.append((row, col))

    while q:
        row, col = q.popleft()
        for r, c in DIR:
            n_r, n_c = row + r, col + c

            if n_r < 0 or n_r >= m or n_c < 0 or n_c >= n or mat[n_r][n_c] != -1:
                continue
            mat[n_r][n_c] = mat[row][col] + 1
            q.append((n_r, n_c))
    return mat


def updateMatrixDP(mat: List[List[int]]) -> List[List[int]]:
    m, n = len(mat), len(mat[0])
    max_ = 10 ** 4
    # top left
    for row in range(m):
        for col in range(n):
            if mat[row][col]:
                top = mat[row - 1][col] if row > 0 else max_
                left = mat[row][col - 1] if col > 0 else max_
                mat[row][col] = min(top, left) + 1

    # bottom right
    for row in range(m - 1, -1, -1):
        for col in range(n - 1, -1, -1):
            if mat[row][col]:
                bottom = mat[row + 1][col] if row < m - 1 else max_
                right = mat[row][col + 1] if col < n - 1 else max_
                mat[row][col] = min(mat[row][col], bottom, right) + 1
    return mat


def sumOfLeftLeaves(root: Optional[TreeNode]) -> int:
    res = 0
    q = collections.deque([root])
    while q:
        curr = q.popleft()
        if curr.left:
            if not curr.left.left and not curr.left.right:
                res += curr.left.val
            q.append(curr.left)
        if curr.right:
            q.append(curr.right)
    return res


def minCostClimbingStairs(cost: List[int]) -> int:
    prev_prev, prev = cost[0], cost[1]

    for i in range(2, len(cost)):
        curr = cost[i] + min(prev, prev_prev)
        prev_prev, prev = prev, curr
    return min(prev_prev, prev)


def reorderLogFiles(logs: List[str]) -> List[str]:
    def get_key(log: str):
        _id, rest = log.split(" ", maxsplit=1)
        return (1,) if rest[0].isdigit() else (0, rest, _id)

    return sorted(logs, key=get_key)


def generateParenthesis(n: int) -> List[str]:
    stack = []
    res = []

    def backtrack(open_n: int, close_n: int):
        if open_n == close_n == n:
            res.append("".join(stack))
            return

        if open_n < n:
            stack.append("(")
            backtrack(open_n + 1, close_n)
            stack.pop()

        if close_n < open_n:
            stack.append(")")
            backtrack(open_n, close_n + 1)
            stack.pop()

    backtrack(0, 0)
    return res


def isPowerOfTwo(n: int) -> bool:
    return n > 0 and math.log2(n) % 1 == 0


def findDisappearedNumbers(nums: List[int]) -> List[int]:
    n = len(nums)
    nums = set(nums)
    return [i for i in range(n) if i not in nums]


def findDisappearedNumbersAlt(nums: List[int]) -> List[int]:
    for num in nums:
        idx = abs(nums[num]) - 1
        nums[idx] = -abs(nums[idx])
    return [i + 1 for i in range(len(nums)) if nums[i] > 0]


def maxProfit(prices: List[int]) -> int:
    n = len(prices)
    l = res = 0
    for r in range(1, n):
        if l > r:
            l = r
        res = max(res, prices[r] - prices[l])
    return res


def productExceptSelf(nums: List[int]) -> List[int]:
    n = len(nums)
    res = [1] * n

    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]

    suffix = 1
    for i in range(n - 1, -1, -1):
        res[i] *= suffix
        suffix *= nums[i]
    return res


def checkRecord(s: str) -> bool:
    absent = 0
    n = len(s)
    for i in range(n):
        if s[i] == "A":
            absent += 1
        if absent > 1:
            return False
        if i < n - 2 and s[i] == s[i + 1] == s[i + 2] == "L":
            return False
    return True


def networkDelayTime(times: List[List[int]], n: int, k: int) -> int:
    adj_list = collections.defaultdict(list)
    for scr, dest, weight in times:
        adj_list[scr].append((dest, weight))

    seen = set()
    min_heap = [(0, k)]
    res = 0
    while min_heap:
        weight, node = heapq.heappop(min_heap)
        if node in seen:
            continue
        seen.add(node)
        res = weight

        for nei_node, nei_weight in adj_list[node]:
            if nei_node not in seen:
                heapq.heappush(min_heap, (weight + nei_weight, nei_node))
    return res if len(seen) == n else -1


def relativeSortArray(arr1: List[int], arr2: List[int]) -> List[int]:
    elem_to_idx = {elem: idx for idx, elem in enumerate(arr2)}
    return sorted(arr1, key=lambda x: elem_to_idx.get(x, 1000 + x))


def numRollsToTarget(n: int, k: int, target: int) -> int:
    memo = {}
    MOD = 10 ** 9 + 7

    def dp(curr_n, val) -> int:
        if curr_n == n:
            return 1 if val == target else 0
        if (curr_n, val) in memo:
            return memo[(curr_n, val)]
        res = 0
        for face in range(1, k + 1):
            res += dp(curr_n + 1, val + face)
        memo[(curr_n, val)] = res % MOD
        return memo[(curr_n, val)]

    return dp(0, 0)


def restoreString(s: str, indices: List[int]) -> str:
    # cyclic sort
    s = list(s)
    n = len(s)
    for i in range(n):
        while indices[i] != i:
            val = indices[i]
            indices[i], indices[val] = indices[val], indices[i]
            s[i], s[val] = s[val], s[i]
    return ''.join(s)


def minTimeToVisitAllPoints(points: List[List[int]]) -> int:
    n = len(points)
    res = 0
    for i in range(1, n):
        res += max(abs(points[i][0] - points[i - 1][0]), abs(points[i][1] - points[i - 1][1]))
    return res
