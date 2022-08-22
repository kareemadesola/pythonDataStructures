import collections


def can_sum(target_sum: int, numbers) -> bool:
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


def test_can_sum():
    print(can_sum(22, [5, 7]))
