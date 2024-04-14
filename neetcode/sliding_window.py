from collections import Counter, deque
from typing import List


def maxProfit(prices: List[int]) -> int:
    min_so_far = prices[0]
    res = 0
    n = len(prices)
    for i in range(1, n):
        min_so_far = min(min_so_far, prices[i])
        res = max(res, prices[i] - min_so_far)
    return res


def maxProfitAlt(prices: List[int]) -> int:
    # sliding window solution
    l, n = 0, len(prices)
    res = 0
    for r in range(1, n):
        if prices[r] < prices[l]:
            l = r
        res = max(res, prices[r] - prices[l])
    return res


def lengthOfLongestSubstring(s: str) -> int:
    seen = set()
    l, n = 0, len(s)
    res = 0
    for r in range(n):
        while s[r] in seen:
            seen.remove(s[l])
            l += 1
        seen.add(s[r])
        res = max(res, r - l + 1)
    return res


def lengthOfLongestSubstringAlt(s: str) -> int:
    char_to_idx = {}
    res = l = 0
    for r, val in enumerate(s):
        if val in char_to_idx and char_to_idx[val] >= l:
            l = char_to_idx[val] + 1
        char_to_idx[val] = r
        res = max(res, r - l + 1)
    return res


def characterReplacement(s: str, k: int) -> int:
    char_to_cnt = {}

    l = res = 0
    for r, char in enumerate(s):
        char_to_cnt[char] = char_to_cnt.get(char, 0) + 1
        while r - l + 1 - max(char_to_cnt.values()) > k:
            char_to_cnt[s[l]] -= 1
            l += 1
        res = max(res, r - l + 1)
    return res


def characterReplacementAlt(s: str, k: int) -> int:
    char_to_cnt = {}

    l = res = max_f = 0
    for r, char in enumerate(s):
        char_to_cnt[char] = char_to_cnt.get(char, 0) + 1
        max_f = max(max_f, char_to_cnt[char])

        while r - l + 1 - max_f > k:
            char_to_cnt[s[l]] -= 1
            l += 1
        res = max(res, r - l + 1)
    return res


def checkInclusion(s1: str, s2: str) -> bool:
    if len(s1) > len(s2):
        return False
    s1_cnt, s2_cnt = [0] * 26, [0] * 26
    for i in range(len(s1)):
        s1_cnt[ord(s1[i]) - ord('a')] += 1
        s2_cnt[ord(s2[i]) - ord('a')] += 1

    matches = 0
    for i in range(26):
        matches += (1 if s1_cnt[i] == s2_cnt[i] else 0)

    l = 0
    for r in range(len(s1), len(s2)):
        if matches == 26:
            return True
        index = ord(s2[r]) - ord('a')
        s2_cnt[index] += 1
        if s1_cnt[index] == s2_cnt[index]:
            matches += 1
        elif s1_cnt[index] + 1 == s2_cnt[index]:
            matches -= 1

        index = ord(s2[l]) - ord('a')
        s2_cnt[index] -= 1
        if s1_cnt[index] == s2_cnt[index]:
            matches += 1
        elif s1_cnt[index] - 1 == s2_cnt[index]:
            matches -= 1
        l += 1
    return matches == 26


def minWindow(s: str, t: str) -> str:
    m, n = len(s), len(t)
    if m < n:
        return ''
    cnt_s, cnt_t = {}, Counter(t)
    have, need = 0, len(cnt_t)
    l = 0
    res = (0, m)
    for r in range(m):
        cnt_s[s[r]] = cnt_s.get(s[r], 0) + 1
        if s[r] in cnt_t and cnt_s[s[r]] == cnt_t[s[r]]:
            have += 1

        while have == need:
            res = (l, r) if r - l < res[1] - res[0] else res
            cnt_s[s[l]] -= 1
            if s[l] in cnt_t and cnt_s[s[l]] < cnt_t[s[l]]:
                have -= 1
            l += 1
    return s[res[0]: res[1] + 1] if res[1] != m else ''


def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    q = deque()
    res = []
    l = 0
    for r in range(len(nums)):
        while q and nums[r] > nums[q[-1]]:
            q.pop()
        q.append(r)

        if l > q[0]:
            q.popleft()

        if r + 1 >= k:
            res.append(q[0])
            l += 1
    return res
