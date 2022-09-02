from typing import List


def island_perimeter(grid: List[List[int]]) -> int:
    res = 0
    len_r, len_c = len(grid), len(grid[0])
    units = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for r in range(len_r):
        for c in range(len_c):
            if grid[r][c] == 1:
                for i, j in units:
                    x, y = r + i, c + j
                    if not 0 <= x < len_r or not 0 <= y < len_c or grid[x][y] == 0:
                        res += 1
    return res


def test_island_perimeter():
    assert island_perimeter([[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0]]) == 16
