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
