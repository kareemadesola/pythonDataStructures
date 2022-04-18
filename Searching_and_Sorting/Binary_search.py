"""You're going to write a binary search function.
You should use an iterative approach - meaning
using loops.
Your function should take two inputs:
a Python list to search through, and the value
you're searching for.
Assume the list only has distinct elements,
meaning there are no repeated values, and
elements are in a strictly increasing order.
Return the index of value, or -1 if the value
doesn't exist in the list."""


def binary_search(input_array, value):
    """Your code goes here."""
    lower_limit = 0
    upper_limit = len(input_array) - 1
    while lower_limit <= upper_limit:
        middle_index = (lower_limit + upper_limit) // 2
        if value == input_array[middle_index]:
            return middle_index
        elif value < input_array[middle_index]:
            upper_limit = middle_index - 1
        else:
            lower_limit = middle_index + 1
    return -1


if __name__ == '__main__':
    test_list = [1, 3, 9, 11, 15, 19, 29]
    test_val1 = 25
    test_val2 = 15
    print(binary_search(test_list, test_val1))
    print(binary_search(test_list, test_val2))
