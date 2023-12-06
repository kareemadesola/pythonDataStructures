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


def totalMoney(n: int) -> int:
    res = 0
    div, mod = divmod(n, 7)

    tmp = 0
    for _ in range(div):
        res += 28 + tmp
        tmp += 7

    tmp = div + 1
    for _ in range(mod):
        res += tmp
        tmp += 1
    return res


def totalMoneyAlt(n: int) -> int:
    res = 0
    monday = 1
    while n > 0:
        for day in range(min(n, 7)):
            res += day + monday
        monday += 1
        n -= 7
    return res


def totalMoneyOpt(n: int) -> int:
    k = n // 7
    f = 28
    l = f + (k - 1) * 7
    arith_sum = k * (f + l) // 2

    rem = n % 7
    monday = k + 1
    for day in range(rem):
        arith_sum += monday + day
    return arith_sum
