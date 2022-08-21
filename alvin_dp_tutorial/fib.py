def dib(n):
    if n <= 1: return 1
    return dib(n - 1) + dib(n - 1)


def fib(n):
    # exponential time complexity
    if n <= 2: return 1
    return dib(n - 1) + dib(n - 2)


def fib_memo(n, memo={}):
    # time O(n)
    # space O(n)
    if n in memo:
        return memo[n]
    if n <= 2: return 1
    memo[n] = fib_memo(n - 1) + fib_memo(n - 2)
    return memo[n]


# def fib_memo(n, memo=None):
#     if memo is None:
#         memo = {}
#     if n in memo:
#         return memo[n]
#     if n <= 2: return 1
#     memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
#     return memo[n]
#

def test():
    # print(fib_memo(8000))
    fib_memo(5)
