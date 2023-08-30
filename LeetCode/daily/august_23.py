import heapq
import itertools
from collections import deque, defaultdict, Counter
from typing import List, Optional

from LeetCode.Biweekly.contest_82 import TreeNode
from LeetCode.explore.linked_list import ListNode


def combine(n: int, k: int) -> List[List[int]]:
    # Tue, 01 Aug 2023  01:39:32
    # time O(r * nCr)
    # space O(r)
    return list(itertools.combinations(range(1, n + 1), k))


def combineRC(n: int, k: int) -> List[List[int]]:
    res = []
    comb = []

    def backtrack(start: int):
        if len(comb) == k:
            res.append(comb.copy())
            return
        for i in range(start, n + 1):
            comb.append(i)
            backtrack(i + 1)
            comb.pop()

    backtrack(1)
    return res


def permute(nums: List[int]) -> List[List[int]]:
    return list(itertools.permutations(nums))


def permuteRC(nums: List[int]) -> List[List[int]]:
    def backtrack():
        if len(curr) == n:
            res.append(curr[:])
        for num in nums:
            if num not in curr:
                curr.append(num)
                backtrack()
                curr.pop()

    n = len(nums)
    res = []
    curr = []
    backtrack()
    return res


def letterCombinations(digits: str) -> List[str]:
    n = len(digits)

    def backtrack(i: int):
        if len(curr) == n:
            res.append("".join(curr[:]))
            return
        for c in digit_to_str[digits[i]]:
            curr.append(c)
            backtrack(i + 1)
            curr.pop()

    digit_to_str = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }
    curr = []
    if digits:
        backtrack(0)
    res = []
    return res


def wordBreak(s: str, wordDict: List[str]) -> bool:
    dp = [False] * (len(s) + 1)
    dp[len(s)] = True

    for i in range(len(s) - 1, -1, -1):
        for w in wordDict:
            if s[i : i + len(w)] == w:
                dp[i] = dp[i + len(w)]
            if dp[i]:
                break
    return dp[0]


def generateTrees(n: int) -> List[Optional[TreeNode]]:
    dp = {}

    def generate(l: int, r: int) -> List[Optional[TreeNode]]:
        if l > r:
            return [None]
        if (l, r) in dp:
            return dp[(l, r)]

        res = []
        for val in range(l, r + 1):
            for l_tree in generate(l, val - 1):
                for r_tree in generate(val + 1, r):
                    root = TreeNode(val, l_tree, r_tree)
                    res.append(root)

        dp[(l, r)] = res
        return res

    return generate(1, n)


def numMusicPlaylists(n: int, goal: int, k: int) -> int:
    mod = 10**9 + 7
    dp = {}

    def count(curr_goal, old_songs):
        if (curr_goal, old_songs) in dp:
            return dp[(curr_goal, old_songs)]
        if curr_goal == 0 and old_songs == n:
            return 1
        if curr_goal == 0 or old_songs > n:
            return 0

        # choose new song
        res = (n - old_songs) * count(curr_goal - 1, old_songs + 1)
        if old_songs > k:
            # choose old song
            res += (old_songs - k) * count(curr_goal - 1, old_songs)
        dp[(curr_goal, old_songs)] = res % mod
        return dp[(curr_goal, old_songs)]

    return count(goal, 0)


def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    def binary_search(arr: List[int]) -> bool:
        l, r = 0, len(arr) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if arr[mid] < target:
                l = mid + 1
            elif arr[mid] == target:
                return True
            else:
                r = mid - 1
        return False

    res = []
    for row in matrix:
        res.extend(row)
    return binary_search(res)


def searchMatrixBS(matrix: List[List[int]], target: int) -> bool:
    top, down = 0, len(matrix)

    while top < down:
        mid = top + (down - top) // 2
        if matrix[mid][-1] < target:
            top = mid + 1
        else:
            down = mid

    if top == len(matrix):
        return False
    l, r = 0, len(matrix[0])
    while l < r:
        mid = l + (r - l) // 2
        if matrix[top][mid] < target:
            l = mid + 1
        else:
            r = mid
    return l < len(matrix[top]) and matrix[top][l] == target


def minimizeMax(nums: List[int], p: int) -> int:
    def is_valid(x):
        i = cnt = 0
        while i < len(nums) - 1:
            if abs(nums[i] - nums[i + 1]) <= x:
                cnt += 1
                i += 2
            else:
                i += 1
            if cnt == p:
                return True
        return False

    if p == 0:
        return 0
    l, r = 0, nums[-1] - nums[0]
    nums.sort()
    while l < r:
        mid = l + (r - l) // 2
        if is_valid(mid):
            r = mid
        else:
            l = mid + 1
    return l


def search(nums: List[int], target: int) -> bool:
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return True
        if nums[mid] > nums[l]:  # left sorted
            if nums[l] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        elif nums[mid] < nums[l]:  # right sorted
            if nums[mid] < target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
        else:
            l += 1
    return False


def coinChange(coins: List[int], amount: int) -> int:
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if a - c >= 0:
                dp[a] = min(dp[a], 1 + dp[a - c])
    return dp[amount] if dp[amount] != amount + 1 else -1


def uniquePaths(m: int, n: int) -> int:
    dp = [[1] * n for _ in range(m)]

    for r in range(1, m):
        for c in range(1, n):
            dp[r][c] = dp[r][c - 1] + dp[r - 1][c]
    return dp[m - 1][n - 1]


def uniquePaths1d(m: int, n: int) -> int:
    row = [1] * n

    for _ in range(m - 1):
        new_row = [1] * n
        for j in range(n - 2, -1, -1):
            new_row[j] = new_row[j + 1] + row[j]
        row = new_row

    return row[0]


def uniquePathsWithObstacles(obstacleGrid: List[List[int]]) -> int:
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = {(m - 1, n - 1): 1}

    def dfs(r: int, c: int) -> int:
        if (r, c) in dp:
            return dp[(r, c)]
        if r == m or c == n or obstacleGrid[r][c]:
            return 0
        dp[(r, c)] = dfs(r - 1, c) + dfs(r, c - 1)
        return dp[(r, c)]

    return dfs(0, 0)


def uniquePathsWithObstaclesBU(obs: List[List[int]]) -> int:
    m, n = len(obs), len(obs[0])

    dp = [[0] * n for _ in range(m)]

    if obs[m - 1][n - 1] == 1 or obs[0][0] == 1:
        return 0

    dp[m - 1][n - 1] = 1

    for i in range(m - 2, -1, -1):
        dp[i][n - 1] = dp[i + 1][n - 1] if obs[i][n - 1] == 0 else 0

    for j in range(n - 2, -1, -1):
        dp[m - 1][j] = dp[m - 1][j + 1] if obs[m - 1][j] == 0 else 0

    for i in range(m - 2, -1, -1):
        for j in range(n - 2, -1, -1):
            if obs[i][j] == 0:
                dp[i][j] = dp[i + 1][j] + dp[i][j + 1]

    return dp[0][0]


def uniquePathsWithObstacles1dBU(obsobstacleGrid: List[List[int]]) -> int:
    m, n = len(obsobstacleGrid), len(obsobstacleGrid[0])
    dp = [0] * n

    dp[-1] = 1
    for r in range(m - 1, -1, -1):
        for c in range(n - 1, -1, -1):
            if obsobstacleGrid[r][c]:
                dp[c] = 0
            elif c + 1 < n:
                dp[c] += dp[c + 1]
    return dp[0]


def validPartition(nums: List[int]) -> bool:
    dp = {}

    def dfs(i: int) -> bool:
        if i == len(nums):
            return True
        if i in dp:
            return dp[i]
        res = False
        if i + 1 < len(nums) and nums[i] == nums[i + 1]:
            res = dfs(i + 2)
        if i + 2 < len(nums):
            if (
                nums[i] == nums[i + 1] == nums[i + 2]
                or nums[i] + 2 == nums[i + 1] + 1 == nums[i + 2]
            ):
                res |= dfs(i + 3)
        dp[i] = res
        return res

    return dfs(0)


def findKthLargest(nums: List[int], k: int) -> int:
    nums = [-i for i in nums]
    heapq.heapify(nums)
    for _ in range(k - 1):
        heapq.heappop(nums)
    return -heapq.heappop(nums)


def findKthLargestQuickSelect(nums: List[int], k: int) -> int:
    k = len(nums) - k

    def quick_select(l: int, r: int) -> int:
        pivot, ptr = nums[r], l
        for i in range(l, r):
            if nums[i] <= pivot:
                nums[i], nums[ptr] = nums[ptr], nums[i]
                ptr += 1
        nums[ptr], nums[r] = nums[r], nums[ptr]

        if ptr > k:
            return quick_select(l, ptr - 1)
        if ptr < k:
            return quick_select(ptr + 1, r)
        return nums[ptr]

    return quick_select(0, len(nums) - 1)


def partition(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    l_dummy = l = ListNode()
    r_dummy = r = ListNode()
    curr = head
    while curr:
        if curr.val < x:
            l.next = curr
            l = l.next
        else:
            r.next = curr
            r = r.next
        curr = curr.next
    r.next = None
    l.next = r_dummy.next
    return l_dummy.next


def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    res = []
    q = deque()

    for i in range(k):
        while q and nums[q[-1]] < nums[i]:
            q.pop()
        q.append(i)
    res.append(nums[q[0]])

    for i in range(k, len(nums)):
        if q[0] == i - k:
            q.popleft()
        while q and nums[q[-1]] < nums[i]:
            q.pop()
        q.append(i)
        res.append(nums[q[0]])
    return res


def maximalNetworkRank(n: int, roads: List[List[int]]) -> int:
    max_rank = 0
    adj = defaultdict(set)
    # create adj matrix
    for road in roads:
        adj[road[0]].add(road[1])
        adj[road[1]].add(road[0])

    for node_1 in range(n):
        for node_2 in range(node_1 + 1, n):
            curr_rank = len(adj[node_1]) + len(adj[node_2])
            if node_2 in adj[node_1]:
                curr_rank -= 1
            max_rank = max(max_rank, curr_rank)
    return max_rank


def updateMatrix(mat: List[List[int]]) -> List[List[int]]:
    q = deque()
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    # make cells with 1 -1 and add cell with 0 to q
    x, y = len(mat), len(mat[0])
    for r in range(x):
        for c in range(y):
            if mat[r][c]:
                mat[r][c] = -1
            else:
                q.append((r, c))

    # start bfs
    while q:
        r, c = q.popleft()
        for i, j in directions:
            n_r, n_c = r + i, c + j
            if 0 <= n_r < x and 0 <= n_c < y and mat[n_r][n_c] == -1:
                mat[n_r][n_c] = mat[r][c] + 1
                q.append((n_r, n_c))
    return mat


def repeatedSubstringPattern(s: str) -> bool:
    n = len(s)
    for i in range(1, n // 2 + 1):
        if n % i == 0:
            pattern = s[:i] * (n // i)
            if pattern == s:
                return True
    return False


def repeatedSubstringPatternBetter(s: str) -> bool:
    return s in (s + s)[1:-1]


def convertToTitle(columnNumber: int) -> str:
    res = []
    while columnNumber:
        columnNumber -= 1
        res.append(chr(columnNumber % 26 + ord("A")))
        columnNumber //= 26
    return "".join(res[::-1])


def reorganizeString(s: str) -> str:
    char_to_freq = Counter(s)
    res = []
    for _ in s:
        if not res or char_to_freq.most_common(1)[0][0] != res[-1]:
            res.append(char_to_freq.most_common(1)[0][0])
        elif len(char_to_freq.most_common(2)) == 2:
            res.append(char_to_freq.most_common(2)[1][0])
        else:
            return ""

        char_to_freq[res[-1]] -= 1
        if char_to_freq[res[-1]] < 0:
            return ""
    return "".join(res)


def reorganizeStringAlt(s: str) -> str:
    max_heap = [(-freq, char) for char, freq in Counter(s).items()]
    heapq.heapify(max_heap)
    prev_freq = 0
    prev_char = None

    res = []
    while max_heap:
        freq, char = heapq.heappop(max_heap)
        res.append(char)
        freq += 1

        if prev_freq < 0:
            heapq.heappush(max_heap, (prev_freq, prev_char))

        prev_char = char
        prev_freq = freq
    if len(s) == len(res):
        return "".join(res)
    return ""


def findLongestChain(pairs: List[List[int]]) -> int:
    pairs.sort(key=lambda x: x[1])
    tail = pairs[0][1]
    res = 1
    for pair in pairs[1:]:
        if tail < pair[0]:
            res += 1
            tail = pair[1]
    return res


def canCross(stones: List[int]) -> bool:
    n = len(stones)
    dp = [[-1] * n for _ in range(n)]
    mark = {val: idx for idx, val in enumerate(stones)}

    def dfs(curr_idx: int, prev_steps: int) -> bool:
        if curr_idx == n - 1:
            return True
        if dp[curr_idx][prev_steps] != -1:
            return dp[curr_idx][prev_steps] == 1
        res = False
        for next_step in range(prev_steps - 1, prev_steps + 2):
            if next_step > 0 and (stones[curr_idx] + next_step) in mark:
                res = res or dfs(mark[stones[curr_idx] + next_step], next_step)
        dp[curr_idx][prev_steps] = 1 if res else 0
        return res

    return dfs(0, 0)


def canCrossBU(stones: List[int]) -> bool:
    n = len(stones)
    dp = [[-1] * n for _ in range(n)]
    mark = {val: idx for idx, val in enumerate(stones)}
    dp[0][0] = True

    return dp[0][0] == 1


class MyStack:
    def __init__(self):
        self.q_1 = deque()

    def push(self, x: int) -> None:
        # add element
        self.q_1.append(x)
        n = len(self.q_1)
        while n > 1:
            self.q_1.append(self.q_1.popleft())
            n -= 1

    def pop(self) -> int:
        return self.q_1.popleft()

    def top(self) -> int:
        return self.q_1[0]

    def empty(self) -> bool:
        return not bool(self.q_1)


def bestClosingTime(customers: str) -> int:
    curr_penalty = min_penalty = customers.count("Y")
    earliest_hour = 0

    for idx, val in enumerate(customers):
        if val == "Y":
            curr_penalty -= 1
        else:
            curr_penalty += 1

        if curr_penalty < min_penalty:
            min_penalty = curr_penalty
            earliest_hour = idx + 1
    return earliest_hour
