import unittest
from typing import List


def strong_password_checker(password: str) -> bool:
    return len(password) > 7 and \
           any(char.islower() for char in password) and \
           any(char.isupper() for char in password) and \
           any(char.isdigit() for char in password) and \
           any(char in "!@#$%^&*()-+" for char in password) and \
           all(password[idx] != password[idx + 1] for idx in range(len(password) - 1))


def successful_pairs(spells: List[int], potions: List[int], success: int) -> List[int]:
    res = [0] * 3
    for idx, val in enumerate(spells):
        count = 0
        temp = [val * i for i in potions]
        for i in temp:
            if i >= success:
                count += 1
        res[idx] = count
    return res


class Test(unittest.TestCase):
    def test_strong_password_checker(self):
        self.assertFalse(strong_password_checker("11A!A!Aa"))
