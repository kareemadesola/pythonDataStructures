# import collections
import collections
from typing import List, Dict


def number_of_pairs(nums: List[int]) -> List[int]:
    len_nums = len(nums)
    count, nums = 0, collections.Counter(nums)
    for i in nums:
        count += nums[i] // 2
    return [count, len_nums - 2 * count]


def maximum_sum(nums: List[int]) -> int:
    res = 0
    hash_map: Dict[int, List[int]] = {}

    def get_sum_digits(n: int):
        digit_sum = 0
        while n:
            n, mod = divmod(n, 10)
            digit_sum += mod
        return digit_sum

    # populate hash_table with key value pairs
    for num in nums:
        get_sum = get_sum_digits(num)
        if get_sum in hash_map:
            hash_map[get_sum].append(num)
        else:
            hash_map[get_sum] = [num]

    for key in hash_map:
        temp, hash_map_key_len = 0, len(hash_map[key])
        if hash_map_key_len == 2:
            temp = hash_map[key][0] + hash_map[key][1]
        elif hash_map_key_len > 2:
            temp = sum(sorted(hash_map[key], reverse=True)[:2])

        res = max(res, temp)

    return res if res else -1
