from typing import List


def lengthOfLastWord(s: str) -> int:
    return len(s.strip().split(' ')[-1])


def lengthOfLastWordAlt(s: str) -> int:
    i = len(s) - 1
    while not s[i]:
        i -= 1
    end = i

    while i >= 0 and s[i]:
        i -= 1
    start = i
    return end - start


def isIsomorphic(s: str, t: str) -> bool:
    s_to_t, t_to_s = {}, {}
    n = len(s)
    for i in range(n):
        if s[i] in s_to_t and s_to_t[s[i]] != t[i] \
                or t[i] in t_to_s and t_to_s[t[i]] != s[i]:
            return False
        s_to_t[s[i]] = t[i]
        t_to_s[t[i]] = s[i]
    return True


def exist(board: List[List[str]], word: str) -> bool:
    m, n = len(board), len(board[0])
    path = set()

    def backtrack(r: int, c: int, curr: int) -> bool:
        if curr == len(word):
            return True
        if not 0 <= r < m or not 0 <= c < n \
                or (r, c) in path or board[r][c] != word[curr]:
            return False
        path.add((r, c))
        res = backtrack(r + 1, c, curr + 1) \
            or backtrack(r - 1, c, curr + 1) \
            or backtrack(r, c + 1, curr + 1) \
            or backtrack(r, c - 1, curr + 1)
        path.remove((r, c))
        return res

    for i in range(m):
        for j in range(n):
            if board[i][j] == word[0] and backtrack(i, j, 0):
                return True
    return False
