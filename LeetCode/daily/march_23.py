from typing import List


def sortArray(nums: List[int]) -> List[int]:
    def merge_sort(arr: List[int]):

        if len(arr) <= 1:
            return
        mid = len(arr) // 2
        l = arr[0: mid]
        r = arr[mid:]
        merge_sort(l)
        merge_sort(r)
        arr1_len, arr2_len = len(l), len(r)
        i = j = 0
        while i < arr1_len and j < arr2_len:
            if l[i] < r[j]:
                arr[i + j] = l[i]
                i += 1
            else:
                arr[i + j] = r[j]
                j += 1
        while i < arr1_len:
            arr[i + j] = l[i]
            i += 1
        while j < arr2_len:
            arr[i + j] = r[j]
            j += 1

    merge_sort(nums)
    return nums


def compress(chars: List[str]) -> int:
    idx = idx_ans = 0
    while idx < len(chars):
        curr_char = chars[idx]
        cnt = 0
        while idx < len(chars) and chars[idx] == curr_char:
            idx += 1
            cnt += 1
        chars[idx_ans] = curr_char
        idx_ans += 1
        if cnt != 1:
            for char in str(cnt):
                chars[idx_ans] = char
                idx_ans += 1
    return idx_ans


def strStr(haystack: str, needle: str) -> int:
    return haystack.find(needle)


def findKthPositive(arr: List[int], k: int) -> int:
    arr = set(arr)
    res = 0
    while k:
        res += 1
        if res not in arr:
            k -= 1
    return res
