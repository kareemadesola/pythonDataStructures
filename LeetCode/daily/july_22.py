from typing import List


# 2022-07-1, Fri, 22:23:09
# time O(nlogn) where n is length of boxTypes due to merge sort
# space O(n) Due to merge sort
# https://leetcode.com/problems/maximum-units-on-a-truck/discuss/999125/JavaPython-3-Sort-by-the-units-then-apply-greedy-algorithm.
# For better implementation
def maximum_units(box_types: List[List[int]], truck_size: int) -> int:
    res = i = 0
    box_types.sort(key=lambda x: -x[1])
    while truck_size > 0 and i < len(box_types):
        mn = min(box_types[i][0], truck_size)
        res += mn * box_types[i][1]
        truck_size -= mn
        i += 1
    return res
