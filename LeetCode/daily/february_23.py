import math
from typing import List


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
