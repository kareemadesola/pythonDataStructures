import collections
from typing import List

from LeetCode.daily.july_22 import TreeNode


class MovingAverage:
    def __init__(self, size: int):
        self.mx_size = size
        self.curr_size = 0
        self.curr_sum = 0
        self.data = collections.deque()

    #
    # def next(self, val: int) -> float:
    #     if self.curr_size == self.mx_size:
    #         self.curr_sum -= self.data.popleft()
    #         self.curr_size -= 1
    #     self.data.append(val)
    #     self.curr_sum += val
    #     self.curr_size += 1
    #     return self.curr_sum / self.curr_size

    def next(self, val: int) -> float:
        self.curr_sum += val
        self.data.append(val)

        if self.curr_size < self.mx_size:
            self.curr_size += 1
        else:
            self.curr_sum -= self.data.popleft()
        return self.curr_sum / self.curr_size


def bfs(root: TreeNode, target: TreeNode) -> int:
    """BFS with no cycles guaranteed """
    if not root or not target: return -1
    queue: collections.deque[TreeNode] = collections.deque([root])
    steps = 0

    while queue:
        size = len(queue)
        for _ in range(size):
            curr = queue.popleft()
            if curr == target:
                return steps
            if curr.left: queue.append(curr.left)
            if curr.right: queue.append(curr.right)
        steps += 1
    return -1


def bfs_visited(root: TreeNode, target: TreeNode) -> int:
    """BFS template when cycles can occur e.g. graphs
    or when you need nodes added to the queue
    multiple times"""
    if not root or not target: return -1
    queue: collections.deque[TreeNode] = collections.deque([root])
    steps, visited = 0, set()

    while queue:
        size = len(queue)
        for _ in range(size):
            curr = queue.popleft()
            if curr == target and curr not in visited:
                return steps
            if curr.left:
                queue.append(curr.left)
                visited.add(curr.left)
            if curr.right:
                queue.append(curr.right)
                visited.add(curr.right)
        steps += 1
    return -1


def walls_and_gates(rooms: List[List[int]]) -> None:
    rows, cols, visited, queue = len(rooms), len(rooms[0]), set(), collections.deque()

    def add_room(r, c):
        if r < 0 or c < 0 or r == rows or c == cols or (r, c) in visited or rooms[r][c] == -1:
            return
        queue.append((r, c))
        visited.add((r, c))

    for i in range(rows):
        for j in range(cols):
            if rooms[i][j] == 0:
                queue.append((i, j))
                visited.add((i, j))

    dist = 0
    while queue:
        for _ in range(len(queue)):
            i, j = queue.popleft()
            rooms[i][j] = dist

            add_room(i + 1, j)
            add_room(i - 1, j)
            add_room(i, j + 1)
            add_room(i, j - 1)

        dist += 1


def num_islands(grid: List[List[str]]) -> int:
    if not grid: return 0
    rows, cols, visited, islands = len(grid), len(grid[0]), set(), 0

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == "1" and (i, j) not in visited:
                queue = collections.deque()
                queue.append((i, j))
                visited.add((i, j))
                while queue:
                    # why r, c = queue.popleft() won't work
                    # solved because r, c changes everything
                    row, col = queue.popleft()
                    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        if not 0 <= r < rows or not 0 <= c < cols or (r, c) in visited or grid[r][c] == "0":
                            continue
                        queue.append((r, c))
                        visited.add((r, c))
                islands += 1
    return islands


def test_num_islands():
    assert 1 == num_islands([["1", "1", "1"], ["0", "1", "0"], ["1", "1", "1"]])


# 2022-08-6, Sat, 07:13:11
# time O(10000)
# space O(10000)

def open_lock(deadends: List[str], target: str) -> int:
    visited = set(deadends)
    if '0000' in visited: return -1
    q, res = collections.deque(['0000']), 0

    while q:
        size = len(q)
        for _ in range(size):
            curr = q.popleft()
            if curr == target: return res
            ans = []
            for i in range(4):
                digit_1 = curr[:i] + str((int(curr[i]) + 1) % 10) + curr[i + 1:]
                digit_2 = curr[:i] + str((int(curr[i]) - 1) % 10) + curr[i + 1:]
                if digit_1 not in visited:
                    ans.append(digit_1)
                    visited.add(digit_1)
                if digit_2 not in visited:
                    ans.append(digit_2)
                    visited.add(digit_2)
            q.extend(ans)
        res += 1
    return -1


# 2022-08-10, Wed, 07:32:28
# time O(n**2) - branch ^ depth -
# sqrt(n) ^ 4
# space O(1)
# https://leetcode.com/problems/perfect-squares/discuss/71475/Short-Python-solution-using-BFS
def num_squares(n: int) -> int:
    if n <= 2:
        return n
    breadth = []

    i = 1
    while i * i <= n:
        breadth.append(i * i)
        i += 1

    res, to_check = 0, {n}
    while to_check:
        temp = set()
        for i in to_check:
            for j in breadth:
                if i == j:
                    return res + 1
                if i < j:
                    break
                temp.add(i - j)
        to_check = temp
        res += 1


# 2022-08-11, Thu, 04:56:30
# time O(n)
# space O(1)
def is_valid(s: str) -> bool:
    stack = []
    hash_map = {')': '(', '}': '{', ']': '['}

    if len(s) % 2 == 1: return False
    for paren in s:
        if paren in hash_map:
            if not stack or stack.pop() != hash_map[paren]:
                return False
        else:
            stack.append(paren)
    return not stack


def daily_temperatures_stack_tuple(temperatures: List[int]) -> List[int]:
    res = [0] * len(temperatures)
    stack: List[tuple[int, int]] = []
    for idx, val in enumerate(temperatures):
        while stack and val > stack[-1][1]:
            temp = stack.pop()
            res[temp[0]] = idx - temp[0]
        stack.append((idx, val))
    return res


def daily_temperatures(temperatures: List[int]) -> List[int]:
    len_temp = len(temperatures)
    res = [0] * len_temp
    stack: List[int] = []
    for idx in range(len_temp):
        while stack and temperatures[idx] > temperatures[stack[-1]]:
            temp = stack.pop()
            res[temp] = idx - temp
        stack.append(idx)
    return res


def eval_rpn(tokens: List[str]) -> int:
    stack: List[int] = []
    for token in tokens:
        if token in "+-*/":
            b, a = stack.pop(), stack.pop()
            stack.append(eval(f'int({a}{token}{b})'))
        else:
            stack.append(int(token))
    return stack[0]


def test_eval_rpn():
    assert eval_rpn(["4", "13", "5", "/", "+"]) == 6
