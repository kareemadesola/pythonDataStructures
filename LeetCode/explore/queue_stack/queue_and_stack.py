import collections
from typing import List, Optional

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


def numIslands(grid):
    def dfs(i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1':
            return 0

        grid[i][j] = '#'
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
        return 1

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                count += dfs(i, j)
    return count


# Definition for a Node.
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node: Node) -> Node:
    old_to_new = {}

    def dfs(node: Node):
        if node in old_to_new:
            return old_to_new[node]

        copy = Node(node.val)
        old_to_new[node] = copy
        for neighbour in node.neighbors:
            copy.neighbors.append(dfs(neighbour))
        return copy

    return dfs(node) if node else None


def find_target_sum_ways(nums: List[int], target: int) -> int:
    dp = {}

    def backtrack(i, total):
        if i == len(nums):
            return 1 if total == target else 0
        if (i, total) in dp:
            return dp[(i, total)]
        dp[(i, total)] = backtrack(i + 1, total + nums[i]) + \
                         backtrack(i + 1, total - nums[i])
        return dp[(i, total)]

    return backtrack(0, 0)


def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    res = []

    def dfs_inorder(node: Optional[TreeNode]):
        if not node: return
        dfs_inorder(node.left)
        res.append(node.val)
        dfs_inorder(node.right)

    dfs_inorder(root)
    return res


def inorder_traversal_stack(root: Optional[TreeNode]) -> List[int]:
    stack, res = [], []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        temp = stack.pop()
        res.append(temp.val)
        root = temp.right

    return res


class MyQueue:

    def __init__(self):
        self.output = []
        self.input = []

    def push(self, x: int) -> None:
        self.input.append(x)

    def pop(self) -> int:
        self.move()
        return self.output.pop()

    def peek(self) -> int:
        self.move()
        return self.output[-1]

    def empty(self) -> bool:
        return not self.output and not self.input

    def move(self) -> None:
        if not self.output:
            while self.input:
                self.output.append(self.input.pop())


class MyStack:

    def __init__(self):
        self.queue = collections.deque()

    def push(self, x: int) -> None:
        self.queue.append(x)
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())

    def pop(self) -> int:
        return self.queue.popleft()

    def top(self) -> int:
        return self.queue[0]

    def empty(self) -> bool:
        return not self.queue


def decode_string(s: str) -> str:
    stack = []
    curr: List[str] = []
    num = 0
    for char in s:
        if char == '[':
            stack.append((curr, num))
            curr, num = [], 0
        elif char == "]":
            prev, prev_num = stack.pop()
            curr = prev + curr * prev_num
        elif char.isdigit():
            num = 10 * num + int(char)
        else:
            curr.append(char)
    return ''.join(curr)


def flood_fill_dfs(image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
    mx_r, mx_c = len(image), len(image[0])
    prev_color = image[sr][sc]

    def dfs(r: int, c: int):
        if 0 <= r < mx_r and 0 <= c < mx_c and image[r][c] == prev_color:
            image[r][c] = color
        dfs(r - 1, c)
        dfs(r + 1, c)
        dfs(r, c - 1)
        dfs(r, c + 1)

    if image[sr][sc] != color: dfs(sr, sc)
    return image


def flood_fill_bfs(image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
    if image[sr][sc] == color:
        return image
    q = collections.deque([(sr, sc)])
    mx_r, mx_c = len(image), len(image[0])
    prev_color = image[sr][sc]
    while q:
        r, c = q.popleft()
        if 0 <= r < mx_r and 0 <= c < mx_c and image[r][c] == prev_color:
            image[r][c] = color
            q.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
    return image


def update_matrix_bfs(mat: List[List[int]]) -> List[List[int]]:
    q = collections.deque()
    mx_r, mx_c = len(mat), len(mat[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for r in range(mx_r):
        for c in range(mx_c):
            if mat[r][c] == 0:
                q.append((r, c))
            else:
                mat[r][c] = -1

    while q:
        r, c = q.popleft()
        for i, j in directions:
            nr, nc = r + i, c + j
            if 0 <= nr < mx_r and 0 <= nc < mx_c and mat[nr][nc] == -1:
                mat[nr][nc] = mat[r][c] + 1
                q.append((nr, nc))
    return mat


def can_visit_all_rooms(rooms: List[List[int]]) -> bool:
    stack = [0]
    seen = set(stack)
    while stack:
        i = stack.pop()
        for j in rooms[i]:
            if j not in seen:
                stack.append(j)
                seen.add(j)
                if len(seen) == len(rooms): return True
    return len(seen) == len(rooms)
