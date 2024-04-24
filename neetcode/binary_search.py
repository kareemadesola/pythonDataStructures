from typing import List


def search(nums: List[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] < target:
            l = mid + 1
        elif nums[mid] > target:
            r = mid - 1
        else:
            return mid
    return -1


def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    l, r = 0, len(matrix)
    while l < r:
        mid = l + (r - l) // 2
        if matrix[mid][-1] < target:
            l = mid + 1
        else:
            r = mid
    if l == len(matrix):
        return False

    row_idx = l
    l, r = 0, len(matrix[0])
    while l <= r:
        mid = l + (r - l) // 2
        if matrix[row_idx][mid] < target:
            l = mid + 1
        elif matrix[row_idx][mid] > target:
            r = mid - 1
        else:
            return True
    return False
