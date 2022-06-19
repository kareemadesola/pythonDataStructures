import unittest
from typing import Set


def greatest_letter(s: str) -> str:
    hash_set: Set[str] = set(s)
    dct = {}
    for char in hash_set:
        if char.isupper() and chr(ord(char) + 32) in hash_set:
            dct[ord(char) + 32] = char
    return '' if not dct else dct[max(dct)]


def minimum_numbers(num: int, k: int) -> int:
    if k > num: return 0
    res = 0
    tmp = k
    # hash_set = set()
    while num - tmp > tmp:
        if (num - tmp) % 10 == k:
            # hash_set.add((tmp, num - tmp))
            res += 1
        tmp += 10
    return res if res else -1


class Test(unittest.TestCase):
    def test_minimum_numbers(self):
        self.assertEqual(2, minimum_numbers(58, 9))
        self.assertEqual(-1, minimum_numbers(37, 2))
