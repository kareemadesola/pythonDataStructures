from typing import List


def fill_cups(amount: List[int]) -> int:
    count = 0
    while amount[0] or amount[1] or amount[2]:
        if amount[0] and amount[1]:
            mn = min(amount[0], amount[1])
            amount[0] -= mn
            amount[1] -= mn
            count += mn
        if amount[0] and amount[2]:
            mn = min(amount[0], amount[2])
            amount[0] -= mn
            amount[2] -= mn
            count += mn

        if amount[1] and amount[2]:
            mn = min(amount[0], amount[1])
            amount[1] -= mn
            amount[2] -= mn
            count += mn
        else:
            return count + max(amount)
