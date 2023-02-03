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
