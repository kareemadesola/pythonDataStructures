from typing import List


def calculate_tax(brackets: List[List[int]], income: int) -> float:
    result = 0
    pre = 0
    for a, b in brackets:
        if a >= income:
            return result + (income - pre) * b / 100
        result += (a - pre) * b / 100
        pre = a
    return result
