import math


def gcdOfStrings(str1: str, str2: str) -> str:
    if str1 + str2 != str2 + str1:
        return ''
    mx = math.gcd(len(str1), len(str2))
    return str1[:mx]
