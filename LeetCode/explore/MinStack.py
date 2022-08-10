from typing import List


class MinStack:

    def __init__(self):
        self.data: List[tuple] = []

    def push(self, val: int) -> None:
        self.data.append((val, min(val, self.getMin() if self.data else val)))

    def pop(self) -> None:
        self.data.pop()

    def top(self) -> int:
        return self.data[-1][0]

    def getMin(self) -> int:
        return self.data[-1][1]
