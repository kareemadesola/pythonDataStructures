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
