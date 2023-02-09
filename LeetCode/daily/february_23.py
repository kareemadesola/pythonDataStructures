import collections
import itertools
import math
from typing import List, Optional


def gcdOfStrings(str1: str, str2: str) -> str:
    if str1 + str2 != str2 + str1:
        return ''
    mx = math.gcd(len(str1), len(str2))
    return str1[:mx]


def isAlienSorted(words: List[str], order: str) -> bool:
    # adapted from https://leetcode.com/problems/verifying-an-alien-dictionary/description/

    # time O(N) => N is the total length of characters in words
    # space O(1) => O(26) == O(1)

    # dict comprehension
    order_to_idx = {char: idx for idx, char in enumerate(order)}

    for i in range(len(words) - 1):
        for j in range(len(words[i])):
            # case example 'leet' 'leetcode'
            if j >= len(words[i + 1]):
                return False
            # compare character of adjacent words
            if words[i][j] != words[i + 1][j]:
                if order_to_idx[words[i][j]] > order_to_idx[words[i + 1][j]]:
                    return False
                break
    return True


def convert(s: str, num_rows: int) -> str:
    if num_rows == 1: return s
    # get no of columns
    # num_cols = (no of Sections) * (no of cols per section)
    # no of sections = no of chars/ chars per section
    n = len(s)
    num_cols = math.ceil(n / (2 * num_rows - 2)) * (num_rows - 1)

    # create matrix of size numRoms * num_cols
    matrix: List[List[Optional[str]]] = [[None] * num_cols for _ in range(num_rows)]

    # fill matrix in the required order
    curr_row, curr_col = 0, 0
    curr_string_index = 0

    while curr_string_index < n:
        # move down
        while curr_row < num_rows and curr_string_index < n:
            matrix[curr_row][curr_col] = s[curr_string_index]
            curr_row += 1
            curr_string_index += 1

        curr_row -= 2
        curr_col += 1

        # move up and right also
        while curr_row > 0 and curr_col < num_cols and curr_string_index < n:
            matrix[curr_row][curr_col] = s[curr_string_index]
            curr_row -= 1
            curr_col += 1
            curr_string_index += 1

    res = ''
    for row in matrix:
        res += ''.join(i for i in row if i)
    return res


def check_inclusion(s1: str, s2: str) -> bool:
    s1_len = len(s1)
    s1_counter = collections.Counter(s1)
    s2_counter = collections.Counter()

    i = 0
    for j in range(len(s2)):
        s2_counter[s2[j]] += 1

        if j - i + 1 == s1_len:
            if s1_counter == s2_counter:
                return True

            s2_counter[s2[i]] -= 1
            i += 1
    return False


def findAnagrams(s: str, p: str) -> List[int]:
    # Sun, 05 Feb 2023  10:11:07
    # time O(S) => length of s
    # space O(P)

    res = []
    p_len = len(p)
    p_counter = collections.Counter(p)
    s_counter = collections.Counter()

    i = 0
    for j in range(len(s)):
        s_counter[s[j]] += 1

        if j - i + 1 == p_len:
            if p_counter == s_counter:
                res.append(i)

            s_counter[s[i]] -= 1
            i += 1
    return res


def findAnagramsOptimized(s: str, p: str) -> List[int]:
    res = []
    p_len = len(p)
    p_counter = collections.Counter(p)

    # pre fill excluding the last element
    s_counter = collections.Counter(s[:p_len - 1])

    for i in range(p_len - 1, len(s)):
        s_counter[s[i]] += 1
        if p_counter == s_counter:
            res.append(i - p_len + 1)
        s_counter[s[i - p_len + 1]] -= 1
    return res


def shuffle(nums: List[int], n: int) -> List[int]:
    # Mon, 06 Feb 2023  19:13:26
    # time O(n) => 2n == n
    # space O(n)
    res = []
    for i in range(n):
        res.extend([nums[i], nums[i + n]])
    return res


def shuffle_bitmask(nums: List[int], n: int) -> List[int]:
    for i in range(n, len(nums)):
        nums[i - n] |= nums[i] << 10

    all_1s = 2 ** 10 - 1

    for i in range(n - 1, -1, -1):
        nums[2 * i + 1] = nums[i] >> 10
        nums[2 * i] = nums[i] & all_1s
    return nums


def totalFruit(fruits: List[int]) -> int:
    mx_picked = 0
    basket = {}
    l = 0
    for r in range(len(fruits)):
        basket[fruits[r]] = basket.get(fruits[r], 0) + 1
        while len(basket) > 2:
            basket[fruits[l]] -= 1
            if not basket[fruits[l]]:
                del basket[fruits[l]]
            l += 1

        mx_picked = max(mx_picked, r - l + 1)
    return mx_picked


def jump(nums: List[int]) -> int:
    res = l = r = farthest = 0
    while r < len(nums) - 1:
        for i in range(l, r + 1):
            farthest = max(farthest, i + nums[i])
        l, r = r + 1, farthest
        res += 1
    return res


def distinctNames(ideas: List[str]) -> int:
    # brute force solution
    ideas = set(ideas)
    res = 0
    for idea_a, idea_b in itertools.combinations(ideas, 2):
        if idea_a[0] + idea_b[1:] not in ideas and idea_b[0] + idea_a[1:] not in ideas:
            res += 2
    return res
