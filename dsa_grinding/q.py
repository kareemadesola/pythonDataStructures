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


def island_perimeter_dfs(grid: List[List[int]]) -> int:
    len_r, len_c = len(grid), len(grid[0])
    visited = set()

    def dfs(x: int, y: int) -> int:
        if (x, y) in visited:
            return 0
        if not 0 <= x < len_r or not 0 <= y < len_c or grid[x][y] == 0:
            return 1
        # alternative you can modify the array to a number
        # representing seen
        visited.add((x, y))
        return dfs(x + 1, y) + dfs(x - 1, y) + dfs(x, y + 1) + dfs(x, y - 1)

    for i in range(len_r):
        for j in range(len_c):
            if grid[i][j]:
                return dfs(i, j)
