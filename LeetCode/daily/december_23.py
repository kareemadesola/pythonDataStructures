import heapq
from collections import Counter, defaultdict
from typing import List, Optional

from sortedcontainers import SortedSet

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


def destCity(paths: List[List[str]]) -> str:
    return ({dst for src, dst in paths} - {src for src, dst in paths}).pop()


def destCityAlt(paths: List[List[str]]) -> str:
    source = {src for src, _ in paths}
    # for dst in list(zip(*paths))[1]:
    for path in paths:
        if path[1] not in source:
            return path[1]


def largestGoodInteger(num: str) -> str:
    n = len(num)
    res = 0
    is_good = False
    for i in range(n - 2):
        if num[i] == num[i + 1] == num[i + 2]:
            is_good = True
            res = max(res, int(num[i:i + 3]))
    if is_good:
        return str(res) if res else '000'
    return ''


def minTimeToVisitAllPoints(points: List[List[int]]) -> int:
    n = len(points)
    res = 0
    for i in range(n - 1):
        res += max(abs(points[i + 1][0] - points[i][0]), abs(points[i + 1][1] - points[i][1]))
    return res


def maxProductDifference(nums: List[int]) -> int:
    nums.sort()
    return nums[-1] * nums[-2] - nums[0] * nums[1]


def maxProductDifferenceAlt(nums: List[int]) -> int:
    max_ = second_max = 0
    min_ = second_min = 10 ** 4
    for num in nums:
        if num > max_:
            max_, second_max = num, max_
        else:
            second_max = max(second_max, num)
        # elif num > second_max:
        #     second_max = num
        if num < min_:
            min_, second_min = num, min_
        else:
            second_min = min(second_min, num)
        # elif num < second_min:
        #     second_min = num
    return max_ * second_max - min_ * second_min


def imageSmoother(img: List[List[int]]) -> List[List[int]]:
    m, n = len(img), len(img[0])

    def average(r: int, c: int) -> int:
        exists = []
        if r - 1 >= 0 and c - 1 >= 0:
            exists.append(img[r - 1][c - 1])
        if r - 1 >= 0:
            exists.append(img[r - 1][c])
        if r - 1 >= 0 and c + 1 < n:
            exists.append(img[r - 1][c + 1])
        if c - 1 >= 0:
            exists.append(img[r][c - 1])
        exists.append(img[r][c])
        if c + 1 < n:
            exists.append(img[r][c + 1])
        if r + 1 < m and c - 1 >= 0:
            exists.append(img[r + 1][c - 1])
        if r + 1 < m:
            exists.append(img[r + 1][c])
        if r + 1 < m and c + 1 < n:
            exists.append(img[r + 1][c + 1])
        return sum(exists) // len(exists)

    res = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            res[i][j] = average(i, j)
    return res


def imageSmootherAlt(img: List[List[int]]) -> List[List[int]]:
    m, n = len(img), len(img[0])
    for r in range(m):
        for c in range(n):
            total = count = 0
            for i in range(r - 1, r + 2):
                for j in range(c - 1, c + 2):
                    if 0 <= i < m and 0 <= j < n:
                        total += img[i][j] & 255
                        count += 1
            img[r][c] |= (total // count) << 8

    for r in range(m):
        for c in range(n):
            img[r][c] >>= 8
    return img


def buyChoco(prices: List[int], money: int) -> int:
    prices.sort()
    sum_two_choco = prices[0] + prices[1]
    return money - sum_two_choco if sum_two_choco <= money else money


def buyChocoAlt(prices: List[int], money: int) -> int:
    heapq.heapify(prices)
    res = money - (heapq.heappop(prices) + heapq.heappop(prices))
    return res if res >= 0 else money


def buyChocoOptimal(prices: List[int], money: int) -> int:
    min_ = second_min = 100
    for price in prices:
        if price < min_:
            min_, second_min = price, min_
        elif price < second_min:
            second_min = price
    res = money - (min_ + second_min)
    return res if res >= 0 else money


class FoodRatings:

    def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
        self.cuisine_to_rating_food = defaultdict(SortedSet)
        self.food_to_rating = {}
        self.food_to_cuisine = {}

        for food, cuisine, ratings in zip(foods, cuisines, ratings):
            self.food_to_rating[food] = ratings
            self.food_to_cuisine[food] = cuisine
            self.cuisine_to_rating_food[cuisine].add((-ratings, food))

    def changeRating(self, food: str, newRating: int) -> None:
        rating = self.food_to_rating[food]
        cuisine = self.food_to_cuisine[food]

        self.cuisine_to_rating_food[cuisine].remove((-rating, food))
        self.cuisine_to_rating_food[cuisine].add((-newRating, food))
        self.food_to_rating[food] = newRating

    def highestRated(self, cuisine: str) -> str:
        return self.cuisine_to_rating_food[cuisine][0][1]


class Food:
    def __init__(self, rating: int, food: str) -> None:
        self.rating = rating
        self.food = food

    def __lt__(self, other: 'Food') -> bool:
        if self.rating == other.rating:
            return self.food < other.food
        return self.rating > other.rating


class FoodRatingsAlt:

    def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
        self.cuisine_to_rating_food = defaultdict(list)
        self.food_to_rating = {}
        self.food_to_cuisine = {}

        for food, cuisine, rating in zip(foods, cuisines, ratings):
            self.food_to_rating[food] = rating
            self.food_to_cuisine[food] = cuisine
            heapq.heappush(self.cuisine_to_rating_food[cuisine], Food(rating, food))

    def changeRating(self, food: str, newRating: int) -> None:
        self.food_to_rating[food] = newRating
        cuisine = self.food_to_cuisine[food]
        heapq.heappush(self.cuisine_to_rating_food[cuisine], Food(newRating, food))

    def highestRated(self, cuisine: str) -> str:
        highest_rated = self.cuisine_to_rating_food[cuisine][0]
        while highest_rated.rating != self.food_to_rating[highest_rated.food]:
            heapq.heappop(self.cuisine_to_rating_food[cuisine])
            highest_rated = self.cuisine_to_rating_food[cuisine][0]
        return highest_rated.food


def maxWidthOfVerticalArea(points: List[List[int]]) -> int:
    points.sort()
    max_width = 0
    for i in range(len(points) - 1):
        if points[i + 1][0] - points[i][0] > max_width:
            max_width = points[i + 1][0] - points[i][0]
    return max_width


def maxScore(s: str) -> int:
    left = 0 if s[0] == '1' else 1
    right = s.count('1', 1)
    res = left + right

    for i in range(1, len(s) - 1):
        if s[i] == '1':
            right = right - 1
        else:
            left = left + 1
        res = max(res, left + right)
    return res


def maxScoreAlt(s: str) -> int:
    # zl + or
    # zl + ot - ol
    zeros = ones = 0
    best = - len(s)

    for i in range(len(s) - 1):
        if s[i] == '1':
            ones += 1
        else:
            zeros += 1
        best = max(best, zeros - ones)
    ones += 1 if s[-1] == '1' else 0

    return best + ones


def isPathCrossing(path: str) -> bool:
    seen = {(0, 0)}
    char_to_dir = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}
    curr_x, curr_y = 0, 0
    for char in path:
        x, y = char_to_dir[char]
        curr_x += x
        curr_y += y
        if (curr_x, curr_y) in seen:
            return True
        seen.add((curr_x, curr_y))
    return False
