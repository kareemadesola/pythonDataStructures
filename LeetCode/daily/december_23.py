from typing import List


def arrayStringsAreEqual(word1: List[str], word2: List[str]) -> bool:
    return ''.join(word1) == ''.join(word2)


def numberOfMatches(n: int) -> int:
    return n - 1
