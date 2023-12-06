from typing import List


def arrayStringsAreEqual(word1: List[str], word2: List[str]) -> bool:
    return ''.join(word1) == ''.join(word2)


def numberOfMatches(n: int) -> int:
    return n - 1


def numberOfMatchesAlt(n: int) -> int:
    res = 0
    while n > 1:
        if n % 2:
            n = (n - 1) // 2 + 1
            res += (n - 1) // 2
        else:
            res += n // 2
            n = n // 2
    return res
