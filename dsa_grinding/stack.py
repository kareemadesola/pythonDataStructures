from typing import List


def min_operations(logs: List[str]) -> int:
    count = 0
    for log in logs:
        if log != "../" and log != "./":
            count += 1
        elif count and log == "../":
            count -= 1
    return count
