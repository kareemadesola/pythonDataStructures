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


def canWinNim(n: int) -> bool:
    return n % 4 != 0


def hammingDistance(x: int, y: int) -> int:
    return (x ^ y).bit_count()


def reverseStr(s: str, k: int) -> str:
    s_list = list(s)
    i, n = 0, len(s)
    while i < len(s):
        s_list[i:i + k] = reversed(s_list[i:i + k])
        i += 2 * k
    return ''.join(s_list)


def reverseStrAlt(s: str, k: int) -> str:
    s_list = list(s)
    for i in range(0, len(s), 2 * k):
        s_list[i:i + k] = reversed(s_list[i:i + k])
    return ''.join(s_list)


def criticalConnections(n: int, connections: List[List[int]]) -> List[List[int]]:
    adj_list = defaultdict(list)
    for fro, to in connections:
        adj_list[fro].append(to)
        adj_list[to].append(fro)
    connections = {tuple(sorted([fro, to])) for fro, to in connections}
    rank = [-2] * n

    def dfs(node: int, depth: int) -> int:
        if rank[node] <= depth:
            return rank[node]
        min_back_depth = n
        for neighbor in adj_list[node]:
            if rank[neighbor] == depth - 1:
                continue
            back_depth = dfs(neighbor, depth + 1)
            if back_depth <= depth:
                connections.discard(tuple(sorted([node, neighbor])))
            min_back_depth = min(min_back_depth, back_depth)
        return min_back_depth

    dfs(0, 0)
    return list(connections)
