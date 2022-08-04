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
