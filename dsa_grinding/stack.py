from typing import List


def min_operations(logs: List[str]) -> int:
    count = 0
    for log in logs:
        if count and log == "../":
            count -= 1
        elif log == "./":
            continue
        else:
            count += 1
    return count
