"""
What is the base case?
When can I no longer continue (the laziest attempt)

What is the smallest amount of work I can do in each iteration?
Shrinks the problem space
Smallest unit of work to contribute
"""


def reverse_string(val: str):
    if val == '':
        return ''
    return reverse_string(val[1:]) + val[0]


def test_reverse_string():
    assert 'olleh' == reverse_string('hello')


def is_palindrome(val: str):
    if len(val) == 1 or len(val) == 0:
        return True
    if val[0] == val[-1]:
        return is_palindrome(val[1:-1])
    return False


def test_is_palindrome():
    assert is_palindrome('aba')


def decimal_to_binary(val: int):
    if val == 0:
        return ''
    div, mod = divmod(val, 2)
    return decimal_to_binary(div) + str(mod)


def decimal_to_binary_alt(val: int, res: str):
    if val == 0: return res
    return decimal_to_binary_alt(val // 2, str(val % 2) + res)


def test_decimal_to_binary():
    assert '1100' == decimal_to_binary(12)


def test_decimal_to_binary_alt():
    assert '1100' == decimal_to_binary_alt(12, '')


def sum_natural_numbers(val: int):
    if val == 1: return 1
    return val + sum_natural_numbers(val - 1)


def test_sum_natural_numbers():
    assert 15 == sum_natural_numbers(5)
