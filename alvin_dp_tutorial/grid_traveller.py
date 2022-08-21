def grid_traveller(m: int, n: int) -> int:
    if m == 1 and n == 1: return 1
    if m == 0 or n == 0: return 0
    return grid_traveller(m - 1, n) + grid_traveller(m, n - 1)


def test_grid_traveller():
    print(grid_traveller(3, 3))


def grid_traveller_memo(m: int, n: int, memo={}) -> int:
    # time O(m*n)
    # space O(m*n)
    if (m, n) in memo: return memo[(m, n)]
    if m == 1 and n == 1: return 1
    if m == 0 or n == 0: return 0
    memo[(m, n)] = grid_traveller_memo(m - 1, n) + grid_traveller_memo(m, n - 1)
    return memo[(m, n)]


def test_grid_traveller_memo():
    print(grid_traveller_memo(3, 3))


def grid_traveller_opt(m: int, n: int, memo={}) -> int:
    if (m, n) in memo or (n, m) in memo: return memo[(m, n)] if (m, n) in memo else memo[(n, m)]
    if m == 1 and n == 1: return 1
    if m == 0 or n == 0: return 0
    if (m, n) not in memo or (n, m) not in memo:
        memo[(m, n)] = grid_traveller_opt(m - 1, n) + grid_traveller_opt(m, n - 1)
    return memo[(m, n)] if (m, n) in memo else memo[(n, m)]


def test_grid_traveller_opt():
    print(grid_traveller_opt(3, 3))
