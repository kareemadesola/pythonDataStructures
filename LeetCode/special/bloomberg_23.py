import collections
import random


class UndergroundSystem:
    def __init__(self):
        self.check_in = {}
        self.distance = collections.defaultdict(list)

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.check_in[id] = (stationName, t)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        check_in_station_name, check_in_time = self.check_in.pop(id)
        self.distance[(check_in_station_name, stationName)].append(t - check_in_time)

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        return sum(self.distance[(startStation, endStation)]) / len(
            self.distance[(startStation, endStation)]
        )


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
