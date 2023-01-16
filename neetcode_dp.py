from typing import List


def climb_stairs(n: int) -> int:
    tmp, res = 0, 1
    for _ in range(n):
        tmp, res = res, tmp + res
    return res


def min_cost_climbing_stairs(cost: List[int]) -> int:
    cost.append(0)
    for i in range(len(cost) - 3, -1, -1):
        cost[i] += min(cost[i + 1], cost[i + 2])
    return min(cost[0], cost[1])
