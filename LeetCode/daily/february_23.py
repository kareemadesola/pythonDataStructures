import collections
import heapq
import itertools
import math
from typing import List, Optional, Deque

from LeetCode.daily.july_22 import TreeNode


def gcdOfStrings(str1: str, str2: str) -> str:
    if str1 + str2 != str2 + str1:
        return ''
    mx = math.gcd(len(str1), len(str2))
    return str1[:mx]


def isAlienSorted(words: List[str], order: str) -> bool:
    # adapted from https://leetcode.com/problems/verifying-an-alien-dictionary/description/

    # time O(N) => N is the total length of characters in words
    # space O(1) => O(26) == O(1)

    # dict comprehension
    order_to_idx = {char: idx for idx, char in enumerate(order)}

    for i in range(len(words) - 1):
        for j in range(len(words[i])):
            # case example 'leet' 'leetcode'
            if j >= len(words[i + 1]):
                return False
            # compare character of adjacent words
            if words[i][j] != words[i + 1][j]:
                if order_to_idx[words[i][j]] > order_to_idx[words[i + 1][j]]:
                    return False
                break
    return True


def convert(s: str, num_rows: int) -> str:
    if num_rows == 1: return s
    # get no of columns
    # num_cols = (no of Sections) * (no of cols per section)
    # no of sections = no of chars/ chars per section
    n = len(s)
    num_cols = math.ceil(n / (2 * num_rows - 2)) * (num_rows - 1)

    # create matrix of size numRoms * num_cols
    matrix: List[List[Optional[str]]] = [[None] * num_cols for _ in range(num_rows)]

    # fill matrix in the required order
    curr_row, curr_col = 0, 0
    curr_string_index = 0

    while curr_string_index < n:
        # move down
        while curr_row < num_rows and curr_string_index < n:
            matrix[curr_row][curr_col] = s[curr_string_index]
            curr_row += 1
            curr_string_index += 1

        curr_row -= 2
        curr_col += 1

        # move up and right also
        while curr_row > 0 and curr_col < num_cols and curr_string_index < n:
            matrix[curr_row][curr_col] = s[curr_string_index]
            curr_row -= 1
            curr_col += 1
            curr_string_index += 1

    res = ''
    for row in matrix:
        res += ''.join(i for i in row if i)
    return res


def check_inclusion(s1: str, s2: str) -> bool:
    s1_len = len(s1)
    s1_counter = collections.Counter(s1)
    s2_counter = collections.Counter()

    i = 0
    for j in range(len(s2)):
        s2_counter[s2[j]] += 1

        if j - i + 1 == s1_len:
            if s1_counter == s2_counter:
                return True

            s2_counter[s2[i]] -= 1
            i += 1
    return False


def findAnagrams(s: str, p: str) -> List[int]:
    # Sun, 05 Feb 2023  10:11:07
    # time O(S) => length of s
    # space O(P)

    res = []
    p_len = len(p)
    p_counter = collections.Counter(p)
    s_counter = collections.Counter()

    i = 0
    for j in range(len(s)):
        s_counter[s[j]] += 1

        if j - i + 1 == p_len:
            if p_counter == s_counter:
                res.append(i)

            s_counter[s[i]] -= 1
            i += 1
    return res


def findAnagramsOptimized(s: str, p: str) -> List[int]:
    res = []
    p_len = len(p)
    p_counter = collections.Counter(p)

    # pre fill excluding the last element
    s_counter = collections.Counter(s[:p_len - 1])

    for i in range(p_len - 1, len(s)):
        s_counter[s[i]] += 1
        if p_counter == s_counter:
            res.append(i - p_len + 1)
        s_counter[s[i - p_len + 1]] -= 1
    return res


def shuffle(nums: List[int], n: int) -> List[int]:
    # Mon, 06 Feb 2023  19:13:26
    # time O(n) => 2n == n
    # space O(n)
    res = []
    for i in range(n):
        res.extend([nums[i], nums[i + n]])
    return res


def shuffle_bitmask(nums: List[int], n: int) -> List[int]:
    for i in range(n, len(nums)):
        nums[i - n] |= nums[i] << 10

    all_1s = 2 ** 10 - 1

    for i in range(n - 1, -1, -1):
        nums[2 * i + 1] = nums[i] >> 10
        nums[2 * i] = nums[i] & all_1s
    return nums


def totalFruit(fruits: List[int]) -> int:
    mx_picked = 0
    basket = {}
    l = 0
    for r in range(len(fruits)):
        basket[fruits[r]] = basket.get(fruits[r], 0) + 1
        while len(basket) > 2:
            basket[fruits[l]] -= 1
            if not basket[fruits[l]]:
                del basket[fruits[l]]
            l += 1

        mx_picked = max(mx_picked, r - l + 1)
    return mx_picked


def jump(nums: List[int]) -> int:
    res = l = r = farthest = 0
    while r < len(nums) - 1:
        for i in range(l, r + 1):
            farthest = max(farthest, i + nums[i])
        l, r = r + 1, farthest
        res += 1
    return res


def distinctNames(ideas: List[str]) -> int:
    # brute force solution
    ideas = set(ideas)
    res = 0
    for idea_a, idea_b in itertools.combinations(ideas, 2):
        if idea_a[0] + idea_b[1:] not in ideas and idea_b[0] + idea_a[1:] not in ideas:
            res += 2
    return res


def distinctNamesOptimized(ideas: List[str]) -> int:
    initial_groups = [set() for _ in range(26)]
    for idea in ideas:
        initial_groups[ord(idea[0]) - ord('a')].add(idea[1:])

    res = 0
    for i in range(25):
        for j in range(i + 1, 26):
            num_of_mutual = len(initial_groups[i] & initial_groups[j])

            res += 2 * (len(initial_groups[i]) - num_of_mutual) * len(initial_groups[j]) - num_of_mutual
    return res


def maxDistance(grid: List[List[int]]) -> int:
    # brute force TLE solution
    n = len(grid)
    ones = []
    zeros = []
    for i in range(n):
        for j in range(n):
            if grid[i][j]:
                ones.append((i, j))
            else:
                zeros.append((i, j))
    if not ones or not zeros: return -1
    temp = []
    for x0, y0 in zeros:
        mn = 2 * n - 2
        for x1, y1 in ones:
            mn = min(mn, abs(x0 - x1) + abs(y0 - y1))
        temp.append(mn)
    return max(temp)


def maxDistanceBFS(grid: List[List[int]]) -> int:
    n = len(grid)
    q = collections.deque()

    for r in range(n):
        for c in range(n):
            if grid[r][c]:
                q.append((r, c))

    res = 0
    directions = ((0, -1), (0, 1), (-1, 0), (1, 0))
    while q:
        r, c = q.popleft()
        res = grid[r][c]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if min(nr, nc) >= 0 and max(nr, nc) < n and not grid[nr][nc]:
                q.append((nr, nc))
                grid[nr][nc] = res + 1
    return res - 1 if res > 1 else -1


def shortestAlternatingPaths(n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
    red, blue = collections.defaultdict(list), collections.defaultdict(list)

    # populate adjacency list
    for src, dst in redEdges:
        red[src].append(dst)
    for src, dst in blueEdges:
        blue[src].append(dst)

    q: Deque[tuple[int, int, Optional[str]]] = collections.deque([(0, 0, None)])
    visited = {(0, None)}
    res = [-1] * n
    while q:
        node, length, edge_color = q.popleft()
        if res[node] == -1:
            res[node] = length

        if edge_color != 'RED':
            for nei in red[node]:
                if (nei, 'RED') not in visited:
                    visited.add((nei, 'RED'))
                    q.append((nei, length + 1, 'RED'))

        if edge_color != 'BLUE':
            for nei in blue[node]:
                if (nei, 'BLUE') not in visited:
                    visited.add((nei, 'BLUE'))
                    q.append((nei, length + 1, 'BLUE'))
    return res


def minimumFuelCost(roads: List[List[int]], seats: int) -> int:
    adj = collections.defaultdict(list)
    for src, dst in roads:
        adj[src].append(dst)
        adj[dst].append(src)

    def dfs(node, parent):
        nonlocal res
        passengers = 0
        for child in adj[node]:
            if child != parent:
                p = dfs(child, node)
                passengers += p
                res += math.ceil(p / seats)
        return passengers + 1

    res = 0
    dfs(0, -1)
    return res


def countOdds(low: int, high: int) -> int:
    res = math.ceil((high - low) / 2)
    return res + 1 if (low % 2 and high % 2) else res


def countOddsBetter(low: int, high: int) -> int:
    if not low % 2:
        low += 1
    return (high - low) // 2 + 1


def addToArrayForm(num: List[int], k: int) -> List[int]:
    return list(map(int, str(int(''.join(map(str, num))) + k)))


def addToArrayFormMath(num: List[int], k: int) -> List[int]:
    for i in range(len(num) - 1, -1, -1):
        k, num[i] = divmod(num[i] + k, 10)
    return num if not k else [int(i) for i in str(k)] + num


def maxDepth(root: Optional[TreeNode]) -> int:
    # Thu, 16 Feb 2023  07:19:39
    # time O(N) => N is the total number of nodes in the tree
    # space O(H) => H is height of tree
    def dfs(node: Optional[TreeNode]) -> int:
        if not node: return 0
        return 1 + max(dfs(node.left), dfs(node.right))

    return dfs(root)


def minDiffInBST(self, root: Optional[TreeNode]) -> int:
    # Fri, 17 Feb 2023  18:07:25
    # time O(N) => N = Number of nodes of root
    # space O(H) => H = height of tree
    res = 10 ** 5
    prev: Optional[TreeNode] = None

    def dfs(node: Optional[TreeNode]):
        nonlocal res, prev
        if not node:
            return
        dfs(node.left)
        if prev:
            res = min(res, node.val - prev.val)
        prev = node

        dfs(node.right)

    dfs(root)
    return res


def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    # Sat, 18 Feb 2023  06:34:57
    # time O(N) => N is number of nodes
    # space O(H) => H is height of root
    def dfs(node: Optional[TreeNode]):
        if not node:
            return
        node.left, node.right = node.right, node.left
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return root


def zigzagLevelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    # time O(N) => N is the number of nodes
    # space O(B) => B is the number of root nodes
    q = collections.deque([root])
    i = 0
    res = []
    while q:
        temp = []
        for _ in range(len(q)):
            curr = q.popleft()
            if curr:
                q.append(curr.left)
                q.append(curr.right)
                temp.append(curr.val)
        if temp:
            res.append(temp if not i % 2 else temp[::-1])
        i += 1
    return res


def searchInsert(nums: List[int], target: int) -> int:
    # Mon, 20 Feb 2023  07:31:19
    # time O(log(n)) => n = len(nums)
    # space O(1)
    # bisect left algorithm
    l, r = 0, len(nums)
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid
    return l


def singleNonDuplicate(nums: List[int]) -> int:
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] == nums[mid ^ 1]:
            l = mid + 1
        else:
            r = mid
    return nums[l]


def shipWithinDays(weights: List[int], days: int) -> int:
    def can_ship(cap: int):
        days_needed, curr_cap = 1, cap
        for weight in weights:
            if curr_cap < weight:
                days_needed += 1
                curr_cap = cap
            curr_cap -= weight
        return days_needed <= days

    l = max(weights)
    r = sum(weights) // days + l
    while l < r:
        mid = l + (r - l) // 2
        if can_ship(mid):
            r = mid
        else:
            l = mid + 1
    return l


def findMaximizedCapital(k: int, w: int, profits: List[int], capital: List[int]) -> int:
    max_profit = []
    min_capital = list(zip(capital, profits))
    heapq.heapify(min_capital)

    for _ in range(k):
        while min_capital and min_capital[0][0] <= w:
            _, p = heapq.heappop(min_capital)
            heapq.heappush(max_profit, -p)
        if not max_profit:
            break
        w += -heapq.heappop(max_profit)
    return w


def minimumDeviation(nums: List[int]) -> int:
    min_heap, heap_max = [], 0
    for num in nums:
        tmp = num
        while num % 2 == 0:
            num //= 2
        min_heap.append((num, max(tmp, 2 * num)))
        heap_max = max(heap_max, num)

    res = 10 ** 9
    heapq.heapify(min_heap)
    while len(min_heap) == len(nums):
        num, n_max = heapq.heappop(min_heap)
        res = min(res, heap_max - num)

        if num < n_max:
            heapq.heappush(min_heap, (num * 2, n_max))
            heap_max = max(heap_max, num * 2)
    return res


def minimum_deviation_better(mx_heap: List[int]) -> int:
    res = 10 ** 9
    mx_heap = [-2 * num if num % 2 else -num for num in mx_heap]
    mn = -max(mx_heap)
    heapq.heapify(mx_heap)

    while mx_heap[0] % 2 == 0:
        n_mx = -heapq.heappop(mx_heap)
        res = min(res, n_mx - mn)
        mn = min(mn, n_mx // 2)
        heapq.heappush(mx_heap, -n_mx // 2)
    return min(res, -mx_heap[0] - mn)
