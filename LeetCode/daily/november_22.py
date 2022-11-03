import collections
from typing import List


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


