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
