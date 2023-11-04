import random
from collections import defaultdict, deque
from typing import List, Optional


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


# Definition for a Node.
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child


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


def removeDuplicatesAlt(s: str, k: int) -> str:
    stack = []  # char, count

    for char in s:
        if stack and stack[-1][0] == char:
            stack[-1][1] += 1
        else:
            stack.append([char, 1])
        if stack[-1][1] == k:
            stack.pop()
    res = []
    for char, count in stack:
        res.append(char * count)
    return "".join(res)


def decodeString(s: str) -> str:
    stack = []
    for char in s:
        if char != "]":
            stack.append(char)
        else:
            tmp = []
            while stack[-1] != "[":
                tmp.append(stack.pop())
            stack.pop()
            tmp = "".join(tmp[::-1])
            num = []
            while stack and stack[-1].isdigit():
                num.append(stack.pop())
            num = int("".join(num[::-1]))
            stack.append(num * tmp)
    return "".join(stack)


def flatten(head: "Optional[Node]") -> "Optional[Node]":
    stack: List[Node] = []
    curr = head
    while curr:
        if curr.child:
            if curr.next:
                stack.append(curr.next)
            curr.next = curr.child
            curr.next.prev = curr
            curr.child = None
        if not curr.next and stack:
            curr.next = stack.pop()
            curr.next.prev = curr
        curr = curr.next
    return head


def numIslands(grid: List[List[str]]) -> int:
    m, n = len(grid), len(grid[0])
    count = 0
    DIR = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def bfs(row: int, col: int):
        q = deque([(row, col)])
        while q:
            r, c = q.popleft()
            for u_r, u_c in DIR:
                n_r, n_c = r + u_r, c + u_c
                if 0 <= n_r < m and 0 <= n_c < n and grid[n_r][n_c] == "1":
                    grid[n_r][n_c] = "#"
                    q.append((n_r, n_c))

    for i in range(m):
        for j in range(n):
            if grid[i][j] == "1":
                grid[i][j] = "#"
                bfs(i, j)
                count += 1
    return count


def numIslandsDFS(grid: List[List[str]]) -> int:
    m, n = len(grid), len(grid[0])
    count = 0
    DIR = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def dfs(row: int, col: int):
        for u_r, u_c in DIR:
            n_r, n_c = row + u_r, col + u_c
            if 0 <= n_r < m or 0 <= n_c < n or grid[n_r][n_c] == "1":
                grid[n_r][n_c] = "#"
                dfs(n_r, n_c)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == "1":
                grid[i][j] = "#"
                dfs(i, j)
                count += 1
    return count


class DNode:
    def __init__(self, val, prev=None, next=None):
        self.val: str = val
        self.prev: Optional[DNode] = prev
        self.next: Optional[DNode] = next


class BrowserHistory:
    def __init__(self, homepage: str):
        self.head = self.curr = DNode(homepage)

    def visit(self, url: str) -> None:
        self.curr.next = DNode(url)
        self.curr.next.prev = self.curr
        self.curr = self.curr.next

    def back(self, steps: int) -> str:
        n = steps
        while self.curr.prev and n:
            self.curr = self.curr.prev
            n -= 1
        return self.curr.val

    def forward(self, steps: int) -> str:
        n = steps
        while self.curr.next and n:
            self.curr = self.curr.next
            n -= 1
        return self.curr.val


class BrowserHistoryAlt:
    def __init__(self, homepage: str):
        self.data = [homepage]
        self.currIdx = 0

    def visit(self, url: str) -> None:
        self.data[self.currIdx + 1 :] = [url]
        self.currIdx += 1

    def back(self, steps: int) -> str:
        self.currIdx = max(0, self.currIdx - steps)
        return self.data[self.currIdx]

    def forward(self, steps: int) -> str:
        self.currIdx = min(len(self.data) - 1, self.currIdx + steps)
        return self.data[self.currIdx]
