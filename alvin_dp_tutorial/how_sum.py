def how_sum(target_sum, numbers):
    res = []

    def helper(target, tmp):
        if target < 0:
            return
        if target == 0:
            res.append(tmp[:])
        for num in numbers:
            tmp.append(num)
            helper(target - num, tmp)
            tmp.pop()

    helper(target_sum, [])
    return res


def test_how_sum():
    print(how_sum(7, [2, 3]))
    print(how_sum(300, [7, 14]))
