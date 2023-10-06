from collections import Counter, defaultdict
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


class ListNode:
    def __init__(self, key=-1, val=-1, nxt=None):
        self.key = key
        self.val = val
        self.next = nxt


class MyHashMapBetter:
    def __init__(self):
        self.data = [ListNode() for _ in range(1000)]

    def hash(self, key):
        return key % 1000

    def put(self, key: int, value: int) -> None:
        curr = self.data[self.hash(key)]
        while curr and curr.next:
            if curr.next.key == key:
                curr.next.val = value
                return
            curr = curr.next
        curr.next = ListNode(key, value)

    def get(self, key: int) -> int:
        curr = self.data[self.hash(key)]
        while curr:
            if curr.key == key:
                return curr.val
            curr = curr.next
        return -1

    def remove(self, key: int) -> None:
        curr = self.data[self.hash(key)]
        while curr and curr.next:
            if curr.next.key == key:
                curr.next = curr.next.next
                return
            curr = curr.next


def majorityElement(nums: List[int]) -> List[int]:
    nums_len = len(nums)
    elem_to_freq = Counter(nums)
    res = []
    for elem in elem_to_freq:
        if elem_to_freq[elem] > nums_len // 3:
            res.append(elem)
    return res


def majorityElementConstantSpace(nums: List[int]) -> List[int]:
    count = defaultdict(int)

    for num in nums:
        count[num] += 1

        if len(count) < 3:
            continue

        new_count = defaultdict(int)
        for k, v in count.items():
            if count[k] > 1:
                new_count[k] = v - 1
    n = len(nums) // 3
    return [k for k in count if nums.count(k) > n]
