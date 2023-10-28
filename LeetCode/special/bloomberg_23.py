import random
from collections import defaultdict
from typing import List


class UndergroundSystem:
    def __init__(self):
        self.check_in = {}
        self.distance = {}

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.check_in[id] = (stationName, t)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        check_in_station_name, check_in_time = self.check_in.pop(id)
        if (check_in_station_name, stationName) not in self.distance:
            self.distance[(check_in_station_name, stationName)] = [0, 0]
        self.distance[(check_in_station_name, stationName)][0] += t - check_in_time
        self.distance[(check_in_station_name, stationName)][1] += 1

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        total, count = self.distance[(startStation, endStation)]
        return total / count


# Your UndergroundSystem object will be instantiated and called as such:
# obj = UndergroundSystem()
# obj.checkIn(id,stationName,t)
# obj.checkOut(id,stationName,t)
# param_3 = obj.getAverageTime(startStation,endStation)

"""
This is Sea's API interface.
You should not implement it, or speculate about its implementation
"""


class Sea:
    def hasShips(self, topRight: "Point", bottomLeft: "Point") -> bool:
        return random.choice([True, False])


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


def countShips(sea: "Sea", topRight: "Point", bottomLeft: "Point") -> int:
    def dfs(t_r: Point, b_l: Point) -> int:
        # check for overlapping
        if b_l.x > t_r.x or bottomLeft.y > t_r.y:
            return 0
        # check if sea has ships
        if not sea.hasShips(t_r, b_l):
            return 0
        # base case
        if b_l.x == b_l.y == t_r.x == t_r.y:
            return 1
        # divide and conquer
        res = 0

        m_x, m_y = (b_l.x + t_r.x) // 2, (b_l.y + t_r.y) // 2
        # top right
        res += dfs(t_r, Point(m_x + 1, m_y + 1))
        # bottom right
        res += dfs(Point(t_r.x, m_y), Point(m_x + 1, b_l.y))
        # top left
        res += dfs(Point(m_x, t_r.y), Point(b_l.x, m_y + 1))
        # bottom left
        res += dfs(Point(m_x, m_y), b_l)
        return res

    return dfs(topRight, bottomLeft)


def invalidTransactions(transactions: List[str]) -> List[str]:
    name_to_info = defaultdict(list)
    for trans in transactions:
        name, time, amount, city = trans.split(",")
        name_to_info[name].append((int(time), int(amount), city))

    res = []
    for trans in transactions:
        name, time, amount, city = trans.split(",")
        time, amount = int(time), int(amount)
        if amount > 1000:
            res.append(trans)
            continue
        for t, a, c in name_to_info[name]:
            if city != c and abs(time - t) <= 60:
                res.append(trans)
                break
    return res


def removeDuplicates(s: str, k: int) -> str:
    stack = []

    for char in s:
        stack.append(char)
        if len(stack) >= k and all(elem == stack[-1] for elem in stack[-k::]):
            for _ in range(k):
                stack.pop()
    return "".join(stack)
