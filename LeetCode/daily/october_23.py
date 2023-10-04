from collections import Counter
from math import comb
from typing import List


def reverseWords(s: str) -> str:
    return " ".join(word[::-1] for word in s.split(" "))


def winnerOfGame(colors: str) -> bool:
    global_cnt_a = global_cnt_b = 0
    local_cnt_a = local_cnt_b = 0

    for char in colors:
        if char == "A":
            local_cnt_a += 1
        else:
            global_cnt_a += max(local_cnt_a - 2, 0)
            local_cnt_a = 0
    global_cnt_a += max(local_cnt_a - 2, 0)

    for char in colors:
        if char == "B":
            local_cnt_b += 1
        else:
            global_cnt_b += max(local_cnt_b - 2, 0)
            local_cnt_b = 0
    global_cnt_b += max(local_cnt_b - 2, 0)
    return global_cnt_a > global_cnt_b


def winnerOfGameOptimal(colors: str) -> bool:
    alice = bob = 0
    for i in range(1, len(colors) - 1):
        if colors[i - 1] == colors[i] == colors[i + 1]:
            if colors[i] == "A":
                alice += 1
            else:
                bob += 1
    return alice > bob


def numIdenticalPairs(nums: List[int]) -> int:
    counter = Counter(nums)
    res = 0
    for count in counter.values():
        res += comb(count, 2)
    return res


class MyHashMap:
    def __init__(self):
        self.data = {}

    def put(self, key: int, value: int) -> None:
        self.data[key] = value

    def get(self, key: int) -> int:
        return self.data.get(key, -1)

    def remove(self, key: int) -> None:
        if key in self.data:
            del self.data[key]
