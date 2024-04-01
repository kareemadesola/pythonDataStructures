from typing import List


def maxProfit(prices: List[int]) -> int:
    min_so_far = prices[0]
    res = 0
    n = len(prices)
    for i in range(1, n):
        min_so_far = min(min_so_far, prices[i])
        res = max(res, prices[i] - min_so_far)
    return res


def maxProfitAlt(prices: List[int]) -> int:
    # sliding window solution
    l, n = 0, len(prices)
    res = 0
    for r in range(1, n):
        if prices[r] < prices[l]:
            l = r
        res = max(res, prices[r] - prices[l])
    return res
