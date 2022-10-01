import collections


def can_sum_bfs(target_sum: int, numbers) -> bool:
    bfs = collections.deque([0])
    while bfs:
        curr = bfs.popleft()
        for i in numbers:
            total = curr + i
            if total == target_sum:
                return True
            if total > target_sum:
                continue
            bfs.append(total)
    return False


# def test_can_sum_bfs():
#     print(can_sum_bfs(22, [5, 7]))


def can_sum(target_sum, numbers) -> bool:
    # time O(numbers**(target_sum/min(numbers))
    # space O(target_sum/min(numbers)
    if target_sum == 0: return True
    if target_sum < 0: return False

    for number in numbers:
        if can_sum(target_sum - number, numbers): return True
    return False


def test_can_sum():
    # assert can_sum(7, [2, 3])
    assert can_sum(7, [5, 3, 4, 7])
    # assert not can_sum(7, [2, 4, ])
    # assert can_sum(8, [2, 3])
    # assert not can_sum(300, [7, 14])
