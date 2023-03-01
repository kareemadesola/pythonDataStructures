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
