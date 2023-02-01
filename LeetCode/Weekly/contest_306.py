from typing import List


def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
    n = len(grid)
    res = [[0] * (n - 2) for _ in range(n - 2)]
    for i in range(n - 2):
        for j in range(n - 2):
            # res[i][j] = max(grid[i][j], grid[i][j + 1], grid[i][j + 2], grid[i + 1][j], grid[i + 1][j + 1],
            #                 grid[i + 1][j + 2],
            #                 grid[i + 2][j], grid[i + 2][j + 1], grid[i + 2][j + 2])
            res[i][j] = max(grid[x][y] for x in range(i, i + 3) for y in range(j, j + 3))
    return res


# def edge_score(edges: List[int]) -> int:
#     # TLE
#     edge_score_dict = collections.defaultdict(int)
#     for idx, val in enumerate(edges):
#         edge_score_dict[val] += idx
#
#     return min(k for k, v in edge_score_dict.items() if v == max(edge_score_dict.values()))

# def edge_score(edges: List[int]) -> int:
#     scores, res = [0] * len(edges), 0
#     for idx, val in enumerate(edges):
#         scores[val] += idx
#
#     for idx, val in enumerate(scores):
#         if val > scores[res]:
#             res = idx
#     return res

# best
def edge_score(edges: List[int]) -> int:
    scores = [0] * len(edges)
    for idx, val in enumerate(edges):
        scores[val] += idx
    return scores.index(max(scores))


def count_special_numbers(n: int) -> int:
    res = 0
    for i in range(1, n + 1):
        if len(str(i)) == len(set(str(i))):
            res += 1
    return res
