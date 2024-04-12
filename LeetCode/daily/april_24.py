from collections import Counter, deque
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


def maxDepth(s: str) -> int:
    res = tmp = 0
    for char in s:
        if char == '(':
            tmp += 1
            res = max(res, tmp)
        elif char == ')':
            tmp -= 1
    return res


def makeGood(s: str) -> str:
    n = len(s)
    stack = [s[0]]
    for i in range(1, n):
        if stack and stack[-1].lower() == s[i].lower() and stack[-1] != s[i]:
            stack.pop()
        else:
            stack.append(s[i])
    return ''.join(stack)


def checkValidString(s: str) -> bool:
    left_min = left_max = 0
    for char in s:
        if char == '(':
            left_min += 1
            left_max += 1
        elif char == ')':
            left_min -= 1
            left_max -= 1
        else:
            left_min -= 1
            left_max += 1
        if left_max < 0:
            return False
        if left_min < 0:  # (*)(
            left_min = 0
    return left_min == 0


def minRemoveToMakeValid(s: str) -> str:
    s_list = []
    diff = 0
    for char in s:
        if char == '(':
            s_list.append(char)
            diff += 1
        elif char == ')':
            if diff - 1 >= 0:
                s_list.append(char)
                diff -= 1
        else:
            s_list.append(char)
    res = []
    for char in s_list[::-1]:
        if char == '(' and diff:
            diff -= 1
        else:
            res.append(char)
    return ''.join(res[::-1])


def countStudents(students: List[int], sandwiches: List[int]) -> int:
    cnt = Counter(students)
    res = len(students)
    for sand in sandwiches:
        if not cnt[sand]:
            return res
        cnt[sand] -= 1
        res -= 1
    return 0


def timeRequiredToBuy(tickets: List[int], k: int) -> int:
    res, n = 0, len(tickets)
    for i in range(n):
        res += min(tickets[i], tickets[k]) if i <= k \
            else min(tickets[i], tickets[k] - 1)
    return res


def deckRevealedIncreasing(deck: List[int]) -> List[int]:
    deck.sort()
    n = len(deck)
    q = deque(range(n))
    res = [0] * n
    for val in deck:
        res[q.popleft()] = val
        if q:
            q.append(q.popleft())
    return res


def trap(height: List[int]) -> int:
    l, r = 0, len(height) - 1
    res = l_max = r_max = 0
    while l <= r:
        if l_max < r_max:
            l_max = max(l_max, height[l])
            res += l_max - height[l]
            l += 1
        else:
            r_max = max(r_max, height[r])
            res += r_max - height[r]
            r -= 1
    return res
