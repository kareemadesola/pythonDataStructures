from typing import List


# time O(k) where k is set list
# space O(k)
def contains_duplicate(nums: List[int]) -> bool:
    a = set()
    for i in nums:
        if i in a:
            return True
        a.add(i)


# time best, average and worst O(N) where N is list length
# space O(k)
def contains_duplicate_pythonic(nums: List[int]) -> bool:
    return not len(nums) != len(set(nums))


# time O(N) N is length of nums
# space O(k) k is the extra space - set
def single_number_set(nums: List[int]) -> int:
    return 2 * sum(set(nums)) - sum(nums)


# time O(N)
# space O(1)
def single_number_xor(nums: List[int]) -> int:
    res = 0
    for i in nums:
        res ^= i
    return res


# time O(max(nums1, nums2))
# space O(nums1)
def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
    return list(set(nums1).intersection(nums2))


# time 0(log n)
# space O(log n)
def is_happy_set(n: int) -> bool:
    def get_next(div) -> int:
        total_sum = 0
        while div != 0:
            div, mod = divmod(div, 10)
            total_sum += mod ** 2
        return total_sum

    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_next(n)

    return n == 1


# time O(log n)
# space O(1)
def is_happy_tortoise_hare(n: int) -> bool:
    def get_next(div: int) -> int:
        total_sum = 0
        while div != 0:
            div, mod = divmod(div, 10)
            total_sum += mod ** 2
        return total_sum

    slow = n
    fast = get_next(n)
    while n != 1 and slow != fast:
        slow = get_next(slow)
        fast = get_next(get_next(fast))
    return n == 1


# time O(log n)
# space O(1)
def is_happy_hardcoded_cycle(n: int) -> bool:
    cycle_members = {4, 16, 37, 58, 89, 145, 42, 20}

    def get_next(div: int) -> int:
        total_sum = 0
        while div != 0:
            div, mod = divmod(div, 10)
            total_sum += mod ** 2
        return total_sum

    while n != 1 and n not in cycle_members:
        n = get_next(n)

    return n == 1


# time O(log n)
# space O(1)
def is_happy_hardcoded_modified(n: int) -> bool:
    def get_next(div: int) -> int:
        total_sum = 0
        while div != 0:
            div, mod = divmod(div, 10)
            total_sum += mod ** 2
        return total_sum

    while n != 1 and n != 4:
        n = get_next(n)

    return n == 1
