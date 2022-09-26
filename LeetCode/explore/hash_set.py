from typing import List


class HashSetArray:
    """Hash set design using list"""

    def __init__(self):
        self.data = []

    #   time O(n) space O(1) where n is the len_ of the array
    def add(self, key: int) -> None:
        if not self.contains(key):
            self.data.append(key)

    # time O(n) space O(1)
    def remove(self, key: int) -> None:
        if self.contains(key):
            self.data.remove(key)

    # time O(n) space O(1)
    def contains(self, key: int) -> bool:
        return key in self.data


class HashSetBit:
    """Hash set design using int bit"""

    def __init__(self):
        self.data = 0

    def add(self, key: int) -> None:
        self.data |= 1 << key

    def remove(self, key: int) -> None:
        self.data &= ~(1 << key)

    def contains(self, key: int) -> bool:
        return bool(self.data & (1 << key))


class HashSetBoolArray:
    """Hash set design using bool array"""

    def __init__(self):
        self.data = [False] * (pow(10, 6) + 1)

    # time O(1)
    def add(self, key: int) -> None:
        if not self.contains(key):
            self.data[key] = True

    # time O(1)
    def remove(self, key: int) -> None:
        if self.contains(key):
            self.data[key] = False

    # time O(1)
    def contains(self, key: int) -> bool:
        return self.data[key]


class HashSet:
    """Hash set design using hash function"""

    def __init__(self):
        self.size = 1000
        self.data: List[List[int]] = [[] for _ in range(self.size)]

    # O(k)
    def add(self, key: int) -> None:
        idx = self.hash_function(key)
        if not self.contains(key):
            self.data[idx].append(key)

    # O(k)
    def remove(self, key: int) -> None:
        idx = self.hash_function(key)
        if self.contains(key):
            self.data[idx].remove(key)

    # O(k) where k = N/len_ N is the number of elements
    def contains(self, key: int) -> bool:
        idx = self.hash_function(key)
        return key in self.data[idx]

    def hash_function(self, key: int) -> int:
        return key % self.size


if __name__ == '__main__':
    a = HashSetArray()
    a.data = 123
    print(a.data)
