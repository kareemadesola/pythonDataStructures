import collections
import random
from typing import List, Optional

from LeetCode.daily.june_22 import TreeNode


def find_ball(grid: List[List[int]]) -> List[int]:
    m, n = len(grid), len(grid[0])

    def helper(i: int) -> int:
        for j in range(m):
            i2 = i + grid[j][i]
            if i2 < 0 or i2 >= n or grid[j][i2] != grid[j][i]:
                return -1
            i = i2
        return i

    return [helper(i) for i in range(n)]


def min_mutation(start: str, end: str, bank: List[str]) -> int:
    q = collections.deque([(start, 0)])
    seen = {start}

    while q:
        node, steps = q.popleft()
        if node == end:
            return steps
        for c in 'ACGT':
            for i in range(len(node)):
                neighbour = node[:i] + c + node[i + 1:]
                if neighbour not in seen and neighbour in bank:
                    q.append((neighbour, steps + 1))
                    seen.add(neighbour)
    return -1


def longest_palindrome(words: List[str]) -> int:
    hash_map = collections.defaultdict(int)
    unpaired = res = 0
    for word in words:
        if word[0] == word[1]:
            if hash_map[word] > 0:
                hash_map[word] -= 1
                unpaired -= 1
                res += 4
            else:
                hash_map[word] += 1
                unpaired += 1
        else:
            if hash_map[word[::-1]] > 0:
                hash_map[word[::-1]] -= 1
                res += 4
            else:
                hash_map[word] += 1
    return res + 2 if unpaired else res


def reverse_vowels(s: str) -> str:
    s, vowels = list(s), 'aeiouAEIOU'
    l, r = 0, len(s) - 1
    while l < r:
        if s[l] in vowels and s[r] in vowels:
            s[l], s[r] = s[r], s[l]
            l += 1
            r -= 1
        elif s[l] in vowels and s[r] not in vowels:
            r -= 1
        elif s[l] not in vowels and s[r] in vowels:
            l += 1
        else:
            l += 1
            r -= 1
    return ''.join(s)


def orderly_queue(s: str, k: int) -> str:
    return ''.join(sorted(s)) if k != 1 else min(s[i:] + s[:i] for i in range(len(s)))


def exist(board: List[List[str]], word: str) -> bool:
    m, n, word_len = len(board), len(board[0]), len(word)
    seen = set()

    def dfs(x, y, len_):
        if len_ == word_len:
            return True
        if not 0 <= x < m or not 0 <= y < n or \
                word[len_] != board[x][y] or (x, y) in seen:
            return False
        seen.add((x, y))

        res = dfs(x - 1, y, len_ + 1) or dfs(x + 1, y, len_ + 1) or \
              dfs(x, y - 1, len_ + 1) or dfs(x, y + 1, len_ + 1)
        seen.remove((x, y))
        return res

    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0): return True
    return False


def maximum69Number(num: int) -> int:
    # Mon, 07 Nov 2022  07:02:07
    # time O(num)
    # space O(num)
    num = str(num)
    for i in range(len(num)):
        if num[i] == '6':
            return int(num[:i] + '9' + num[i + 1:])
    return int(num)


def makeGood(s: str) -> str:
    # Tue, 08 Nov 2022  18:21:50
    # time O(S) where S= len(s)
    # space O(S)
    s: List[str] = list(s)
    i = 0
    while len(s) > 1 and i < len(s) - 1:
        # check if the two characters are different and have the same
        # lowercase form
        if s[i] != s[i + 1] and ord(s[i].lower()) == ord(s[i + 1].lower()):
            s[i:i + 2] = ''
            # for cases like "abBAcC"
            if i != 0: i -= 1

        else:
            i += 1
        return ''.join(s)


class StockSpanner:
    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        # Wed, 09 Nov 2022  17:47:30
        # amortized time O(1) worst time 0(n)
        # amortized space O(1) worst time 0(n)
        res = 1
        while self.stack and self.stack[-1][0] <= price:
            res += self.stack.pop()[1]
        self.stack.append((price, res))
        return res


def remove_duplicates(s: str) -> str:
    # Thu, 10 Nov 2022  14:07:25
    # time O(S) where S is len(s)
    # space O(S)
    stack = []
    for i in s:
        if stack and stack[-1] == i:
            stack.pop()
        else:
            stack.append(i)
    return ''.join(stack)


def remove_duplicates_sorted_array(nums: List[int]) -> int:
    # Fri, 11 Nov 2022  17:41:28
    # time O(N) where N = len(nums)
    # space O(N)
    k = 0
    for j in nums[1:]:
        if nums[k] == j:
            continue
        k += 1
        nums[k] = j
    return k + 1


class MedianFinder:
    # Sat, 12 Nov 2022  19:04:56
    def __init__(self):
        self.data = []

    def addNum(self, num: int) -> None:
        # total time 0(NlogN)
        # bisect_left algorithm
        # time O(log(N)) where N=len(self.data)
        # space O(1)
        l, r = 0, len(self.data)
        while l < r:
            mid = l + (r - l) // 2
            if self.data[mid] < num:
                l = mid + 1
            else:
                r = mid
        self.data.insert(l, num)

    def findMedian(self) -> float:
        # time O(1)
        # space O(1)
        len_ = len(self.data)
        mid = ((len_ // 2) - 1, len_ // 2) if len_ % 2 == 0 else (len_ // 2, len_ // 2)
        return (self.data[mid[0]] + self.data[mid[1]]) / 2


def reverse_words(s: str) -> str:
    # Sun, 13 Nov 2022  16:30:52
    # time O(S) where S = len(s)
    # space O(S)
    return ' '.join(s.split()[::-1])


def removeStones(points):
    uf = {}

    def find(x):
        if x != uf.setdefault(x, x):
            uf[x] = find(uf[x])
        return uf[x]

    for i, j in points:
        uf[find(i)] = find(~j)
    return len(points) - len({find(x) for x in uf})


def count_nodes(root: Optional[TreeNode]) -> int:
    # Tue, 15 Nov 2022  21:01:03
    # time O(N) where N is the number of nodes present
    # space O(H) H is the height of tree
    if not root:
        return 0
    return 1 + count_nodes(root.left) + count_nodes(root.right)


def guess_number(n: int) -> int:
    def guess(num) -> int:
        """predefined api"""
        return random.randint(-1, 1)

    low, high = 1, n

    while not low > high:
        mid = low + (high - low) // 2
        num_guess = guess(mid)

        # guessed number higher than picked number
        if num_guess == -1:
            high = mid - 1
        elif num_guess == 1:
            low = mid + 1
        else:
            return mid


def compute_area(ax1: int, ay1: int,
                 ax2: int, ay2: int,
                 bx1: int, by1: int,
                 bx2: int, by2: int) -> int:
    # Thu, 17 Nov 2022  06:45:51
    # time O(1)
    # space O(1)
    area_a: int = (ax2 - ax1) * (ay2 - ay1)
    area_b: int = (bx2 - bx1) * (by2 - by1)

    x_overlap = max(min(ax2, bx2) - max(ax1, bx1), 0)
    y_overlap = max(min(ay2, by2) - max(ay1, by1), 0)
    area_overlap: int = x_overlap * y_overlap
    return area_a + area_b - area_overlap


def is_ugly(n: int) -> bool:
    # Fri, 18 Nov 2022  19:22:54
    # time O(n)
    # space O(1)

    # singular point
    if n == 0: return False
    while n % 2 == 0 or n % 3 == 0 or n % 5 == 0:
        if n % 2 == 0:
            n /= 2
        if n % 3 == 0:
            n /= 3
        if n % 5 == 0:
            n /= 5
    return n == 1


def is_ugly_optimal(n: int) -> bool:
    if n <= 0: return False

    def keep_dividing(divisor):
        nonlocal n
        while n % divisor == 0:
            n /= divisor
        return n

    for i in (2, 3, 5):
        n = keep_dividing(i)
    return n == 1
