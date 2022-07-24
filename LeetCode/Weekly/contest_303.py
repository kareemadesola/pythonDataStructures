from typing import List


def repeated_character(s: str) -> str:
    hash_map = {}
    for i in s:
        if i in hash_map:
            return i
        hash_map[i] = 1


def equal_pairs(grid: List[List[int]]) -> int:
    if grid == list(map(list, zip(*grid))):
        return len(grid) ** 2
    count = 0
    col_grid = set()
    for i in zip(*grid):
        if i in col_grid:
            count += 1
        col_grid.add(i)

    for i in grid:
        if tuple(i) in col_grid:
            count += 1
    return count
