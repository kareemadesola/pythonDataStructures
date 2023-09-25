from typing import List


def reverseString(s: List[str]) -> None:
    """
    Do not return anything, modify s in-place instead.
    """

    def dfs(l, r):
        if l >= r:
            return
        s[l], s[r] = s[r], s[l]
        dfs(l + 1, r - 1)

    dfs(0, len(s) - 1)
