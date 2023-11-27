import random
from collections import defaultdict
from typing import List


class UndergroundSystem:

    def __init__(self):
        self.check_in = {}
        self.average_time = {}

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.check_in[id] = (stationName, t)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        check_in_station, check_in_time = self.check_in.pop(id)
        if (check_in_station, stationName) not in self.average_time:
            self.average_time[(check_in_station, stationName)] = [0, 0]
        self.average_time[(check_in_station, stationName)][0] += t - check_in_time
        self.average_time[(check_in_station, stationName)][1] += 1

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        time, freq = self.average_time[(startStation, endStation)]
        return time / freq


# """
# This is Sea's API interface.
# You should not implement it, or speculate about its implementation
# """
class Sea:
    def hasShips(self, topRight: 'Point', bottomLeft: 'Point') -> bool:
        return random.choice([True, False])


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class Solution:
    def countShips(self, sea: 'Sea', topRight: 'Point', bottomLeft: 'Point') -> int:
        def dfs(t_r: Point, b_l: Point) -> int:
            if t_r.x > b_l.x or t_r.y > b_l.y:
                return 0

            if not Sea.hasShips(t_r, b_l):
                return 0

            if t_r.x == t_r.y == b_l.x == b_l.y:
                return 1
            res = 0
            mid_x, mid_y = (t_r.x - b_l.x) // 2, (t_r.y - b_l.y) // 2
            # bottom left
            res += dfs(Point(mid_x, mid_y), b_l)
            # bottom_right
            res += dfs(Point(t_r.x, mid_y), Point(mid_x + 1, b_l.y))
            # top_left
            res += dfs(Point(mid_x, t_r.y), Point(b_l.x, mid_y + 1))
            # top_right
            res += dfs(t_r, Point(mid_x + 1, mid_y + 1))
            return res

        return dfs(topRight, bottomLeft)


def invalidTransactions(transactions: List[str]) -> List[str]:
    name_to_info = defaultdict(list)
    for trans in transactions:
        name, time, amount, city = trans.split(',')
        name_to_info[name].append((int(time), int(amount), city))

    res = []
    for trans in transactions:
        name, time, amount, city = trans.split(',')
        time, amount = int(time), int(amount)
        if amount > 1000:
            res.append(trans)
            continue
        for t, a, c in name_to_info[name]:
            if abs(t - time) <= 60 and c != city:
                res.append(trans)
                break
    return res


def knightDialer(n: int) -> int:
    memo = {}
    MOD = 10 ** 9 + 7
    jumps = [
        [4, 6],
        [6, 8],
        [7, 9],
        [4, 8],
        [0, 3, 9],
        [],
        [0, 1, 7],
        [2, 6],
        [1, 3],
        [2, 4],
    ]

    def dp(remain: int, square: int) -> int:
        if (remain, square) in memo:
            return memo[(remain, square)]
        if remain == 0:
            return 1
        ans = 0
        for next_square in jumps[square]:
            ans = (ans + dp(remain - 1, next_square)) % MOD
        memo[(remain, square)] = ans
        return memo[(remain, square)]

    res = 0
    for square in range(10):
        res = (res + dp(n - 1, square)) % MOD
    return res


def knightDialer_BU(n: int) -> int:
    jumps = [
        [4, 6],
        [6, 8],
        [7, 9],
        [4, 8],
        [0, 3, 9],
        [],
        [0, 1, 7],
        [2, 6],
        [1, 3],
        [2, 4],
    ]
    MOD = 10 ** 9 + 7
    prev_dp = [1] * 10
    for remain in range(1, n):
        dp = [0] * 10
        for square in range(10):
            ans = 0
            for next_square in jumps[square]:
                ans = (ans + prev_dp[next_square]) % MOD
            dp[square] = ans

    res = 0
    for val in prev_dp:
        res = (res + val) % MOD
    return res
