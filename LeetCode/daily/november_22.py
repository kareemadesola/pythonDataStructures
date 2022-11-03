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
    # a count variable contains the number of
    # each word
    count = collections.Counter(words)
    res = 0
    central = False
    for word, count_of_word in count.items():
        if word[0] == word[1]:
            if count_of_word % 2 == 0:
                res += count_of_word
            else:
                res += count_of_word - 1
                central = True
        # consider a pair of non-palindrome words,
        # such that one is the reverse of another
        # word[1] + word[0] is the reversed word
        elif word[0] < word[1]:
            res += 2 * min(count_of_word, count[word[1] + word[0]])
    if central:
        res += 1
    return 2 * res
