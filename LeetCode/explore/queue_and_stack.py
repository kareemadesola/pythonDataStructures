import collections
from typing import List

from LeetCode.daily.july_22 import TreeNode


def num_islands(grid: List[List[str]]) -> int:
    pass


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
