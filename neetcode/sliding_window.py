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
