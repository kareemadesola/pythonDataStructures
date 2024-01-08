"""
What is the base case?
When can I no longer continue (the laziest attempt)

What is the smallest amount of work I can do in each iteration?
Shrinks the problem space
Smallest unit of work to contribute
"""
from typing import List

from LeetCode.explore.linked_list.linked_list import ListNode


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


"""
Divide and Conquer
1. Divide problem into several smaller sub-problems
    Normally, the sub-problems are similar to the original
    
2. Conquer the sub-problems by solving them recursively
Base case: solve small enough problems by brute force

3. Combine the solution to get a solution to the sub-problems
And finally a solution to the original problem

4. Divide and Conquer algorithms are normally recursive
"""


def binary_search(array: List[int], x: int, start=0, end=None):
    if end is None: end = len(array) - 1

    def helper(left: int, right: int):

        if left > right:
            return -1
        mid = left + (right - left) // 2
        if x == array[mid]: return mid
        if x < array[mid]: return helper(left, mid - 1)
        return helper(mid + 1, right)

    return helper(start, end)


def test_binary_search():
    assert -1 == binary_search([1, 4, 5, 6], 1, 1)
    assert -1 == binary_search([1, 4, 5, 6], 10)


def merge_sort(data: List[int]):
    def helper(left: int, right: int):
        def merge(start_, mid_, end_):
            # build temp array to avoid modifying the original
            # contents
            temp = [0] * (end_ - start_ + 1)
            i, j, k = start_, mid_ + 1, 0

            # while both sub-array have values, then try and
            # merge them in sorted order
            while i <= mid_ and j <= end_:
                if data[i] <= data[j]:
                    temp[k] = data[i]
                    k += 1
                    i += 1
                else:
                    temp[k] = data[j]
                    k += 1
                    j += 1
            # Add the rest of the values from the left
            # sub-array into the result
            while i <= mid_:
                temp[k] = data[i]
                k += 1
                i += 1
            # Add the rest of the values from the left
            # sub-array into the result
            while j <= end_:
                temp[k] = data[j]
                k += 1
                j += 1
            for i in range(start_, end_ + 1):
                data[i] = temp[i - start_]

        if not left < right: return
        mid = left + (right - left) // 2
        helper(left, mid)
        helper(mid + 1, right)
        merge(left, mid, right)

    start, end = 0, len(data) - 1
    helper(start, end)
    return data


def test_merge_sort():
    assert merge_sort([2, 1, -1, 0]) == [-1, 0, 1, 2]


def reverse_linked_list(head: ListNode):
    def helper(curr: ListNode, prev):
        if not curr:
            return prev
        temp = curr.next
        curr.next = prev

        return helper(temp, curr)

    return helper(head, None)


def test_reverse_linked_list():
    a = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
    b = reverse_linked_list(a)
    c = 0
    # while b:
    #     print(b.val)
    #     b = b.next
    #     c += 1
    while a:
        print(a.val)
        a = a.next
        c += 1
    print(c)


def reverse_linked_list_recursion(head: ListNode):
    if not head or not head.next: return head
    p = reverse_linked_list_recursion(head.next)
    head.next.next = head
    head.next = None
    return p


def merge_two_sorted_list(l1: ListNode, l2: ListNode):
    def sorted_merge(a: ListNode, b: ListNode):
        if not a: return b
        if not b: return a
        if a.val < b.val:
            a.next = sorted_merge(a.next, b)
            return a
        else:
            b.next = sorted_merge(a, b.next)
            return b

    return sorted_merge(l1, l2)


def test_merge_two_sorted_list():
    assert merge_two_sorted_list(ListNode(4), ListNode())
