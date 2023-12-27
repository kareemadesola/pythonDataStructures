from collections import defaultdict
from typing import List


def invalidTransactions(transactions: List[str]) -> List[str]:
    name_to_info = defaultdict(list)

    for tran in transactions:
        name, time, amount, city = tran.split(',')
        time, amount = int(time), int(amount)
        name_to_info[name].append((time, amount, city))

    res = []
    for tran in transactions:
        name, time, amount, city = tran.split(',')
        time, amount = int(time), int(amount)

        if amount > 1000:
            res.append(tran)
            continue

        for time_2, amount_2, city_2 in name_to_info[name]:
            if city != city_2 and abs(time_2 - time) <= 60:
                res.append(tran)
                break
    return res


def twoCitySchedCost(costs: List[List[int]]) -> int:
    n = len(costs)
    for i in range(n):
        a, b = costs[i]
        costs[i] = (b - a, a, b)

    costs.sort()
    res = 0
    for i in range(n // 2):
        res += costs[i][2]

    for i in range(n // 2, n):
        res += costs[i][1]
    return res
