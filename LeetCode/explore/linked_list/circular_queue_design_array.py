from typing import List


class MyCircularQueue:
    def __init__(self, k: int):
        self.max_size = k
        self._data: List[int | None] = [None] * k
        self._head = self._tail = -1

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        if self.isEmpty():
            self._head += 1
        self._tail = (self._tail + 1) % self.max_size
        self._data[self._tail] = value
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        self._data[self._head] = None
        self._head = (self._head + 1) % self.max_size
        if self.isEmpty():
            self._head = self._tail = -1
        return True

    def Front(self) -> int:
        return self._data[self._head] if not self.isEmpty() else -1

    def Rear(self) -> int:
        return self._data[self._tail] if not self.isEmpty() else -1

    def isEmpty(self) -> bool:
        return self._data[self._head] is None

    def isFull(self) -> bool:
        return self._data[(self._tail + 1) % self.max_size] is not None
