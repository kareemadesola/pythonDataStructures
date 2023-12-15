from collections import Counter
from typing import List, Optional

from LeetCode.daily.july_22 import TreeNode


def arrayStringsAreEqual(word1: List[str], word2: List[str]) -> bool:
    return ''.join(word1) == ''.join(word2)


def numberOfMatches(n: int) -> int:
    return n - 1


def numberOfMatchesAlt(n: int) -> int:
    res = 0
    while n > 1:
        if n % 2:
            n = (n - 1) // 2 + 1
            res += (n - 1) // 2
        else:
            res += n // 2
            n = n // 2
    return res


def totalMoney(n: int) -> int:
    res = 0
    div, mod = divmod(n, 7)

    tmp = 0
    for _ in range(div):
        res += 28 + tmp
        tmp += 7

    tmp = div + 1
    for _ in range(mod):
        res += tmp
        tmp += 1
    return res


def totalMoneyAlt(n: int) -> int:
    res = 0
    monday = 1
    while n > 0:
        for day in range(min(n, 7)):
            res += day + monday
        monday += 1
        n -= 7
    return res


def totalMoneyOpt(n: int) -> int:
    k = n // 7
    f = 28
    l = f + (k - 1) * 7
    arith_sum = k * (f + l) // 2

    rem = n % 7
    monday = k + 1
    for day in range(rem):
        arith_sum += monday + day
    return arith_sum


def largestOddNumber(num: str) -> str:
    for r in range(len(num) - 1, -1, -1):
        if int(num[r]) % 2:
            return num[:r + 1]
    return ''


def tree2str(root: Optional[TreeNode]) -> str:
    def dfs(node: Optional[TreeNode]):
        if not node:
            return ''
        if not node.left and not node.right:
            return str(node.val)
        if not node.right:
            return f"{node.val}({dfs(node.left)})"
        return f"{node.val}({dfs(node.left)})({dfs(node.right)})"

    return dfs(root)


def inorderTraversal(root: Optional[TreeNode]) -> List[int]:
    res = []

    def dfs(node: Optional[TreeNode]):
        if not node:
            return
        dfs(node.left)
        res.append(node.val)
        dfs(node.right)

    dfs(root)
    return res


def inorderTraversalIter(root: Optional[TreeNode]) -> List[int]:
    res = []
    stack = []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        curr = curr.right
    return res


def transpose(matrix: List[List[int]]) -> List[List[int]]:
    return [list(tup) for tup in zip(*matrix)]


def transposeAlt(matrix: List[List[int]]) -> List[List[int]]:
    m, n = len(matrix), len(matrix[0])
    return [[matrix[j][i] for j in range(m)] for i in range(n)]


def findSpecialInteger(arr: List[int]) -> int:
    cnt = Counter(arr)
    # return max(cnt, key=lambda x: cnt[x])
    return max(cnt, key=cnt.get)


def findSpecialIntegerAlt(arr: List[int]) -> int:
    n = len(arr)
    if n == 1: return arr[0]
    bar = n // 4
    cnt = i = 1

    while i < n:
        while i < n and arr[i] == arr[i - 1]:
            cnt += 1
            i += 1
        if cnt > bar:
            return arr[i - 1]
        else:
            cnt = 1

        i += 1


def findSpecialIntegerAlt0(arr: List[int]) -> int:
    n = len(arr)
    if n == 1: return arr[0]
    bar = n // 4
    cnt = 1
    i = 0

    while i < n - 1:
        while i < n - 1 and arr[i] == arr[i + 1]:
            cnt += 1
            i += 1
        if cnt > bar:
            return arr[i]
        else:
            cnt = 1

        i += 1


def maxProduct(nums: List[int]) -> int:
    max_max = min_max = 0
    for num in nums:
        if num > max_max:
            max_max, min_max = num, max_max
        elif num > min_max:
            min_max = num
    return (max_max - 1) * (min_max - 1)


def numSpecial(mat: List[List[int]]) -> int:
    m, n = len(mat), len(mat[0])
    ones_row = [0] * m
    ones_col = [0] * n

    for i in range(m):
        for j in range(n):
            ones_row[i] += mat[i][j]
            ones_col[j] += mat[i][j]

    res = 0
    for i in range(m):
        for j in range(n):
            if mat[i][j] and ones_row[i] == 1 and ones_col[j] == 1:
                res += 1
    return res


def onesMinusZeros(grid: List[List[int]]) -> List[List[int]]:
    m, n = len(grid), len(grid[0])

    ones_row = [0] * m
    ones_col = [0] * n

    for i in range(m):
        for j in range(n):
            ones_row[i] += grid[i][j]
            ones_col[j] += grid[i][j]

    grid = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            grid[i][j] = 2 * ones_row[i] + 2 * ones_col[j] - m - n
    return grid
