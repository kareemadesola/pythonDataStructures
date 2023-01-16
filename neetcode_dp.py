def climb_stairs(n: int) -> int:
    tmp, res = 0, 1
    for _ in range(n):
        tmp, res = res, tmp + res
    return res
