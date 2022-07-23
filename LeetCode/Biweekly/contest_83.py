import collections
import heapq
from typing import List, Dict


def bestHand(self, ranks: List[int], suits: List[str]) -> str:
    ranks_cntr, suits_cntr = collections.Counter(ranks), collections.Counter(suits)
    if suits_cntr[suits[0]] == 5:
        return 'Flush'
    mx = 0
    for i in ranks_cntr:
        mx = max(mx, ranks_cntr[i])
    if mx >= 3:
        return 'Three of a Kind'
    if mx == 2:
        return 'Pair'
    return 'High Card'


def zeroFilledSubarray(self, nums: List[int]) -> int:
    cnt = l = r = 0
    while r < len(nums):
        temp = 0
        while nums[r] == 0:
            temp += r - l
            cnt += temp
            r += 1
        l = r
        r += 1
    return cnt


class NumberContainers:

    def __init__(self):
        self.data: Dict[int, List] = collections.defaultdict(list)

    def change(self, index: int, number: int) -> None:
        heapq.heappush(self.data[number], index)

    def find(self, number: int) -> int:
        pass
