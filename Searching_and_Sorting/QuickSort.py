"""Implement quick sort in Python
Input a list.
Output a sorted list.
"""


def partition(array, low, high):
    # Element to be placed at right position
    pivot = array[high]

    # Index of smaller element and indicate
    # the right position of pivot found so far
    i = low
    for j in range(low, high):
        if array[j] <= pivot:
            array[i], array[j] = array[j], array[i]
            i += 1
    array[i], array[high] = array[high], array[i]
    return i


def quicksort(array, low, high):
    if low < high:
        pi = partition(array, low, high)
        # print(pi)
        quicksort(array, low, pi - 1)
        quicksort(array, pi + 1, high)
    return array


#
# def partition(array, low, high):
#     pivot = select_pivot(array)
#     array[pivot], array[-1] = array[-1], array[pivot]
#     print(array[-1])
#     for value in array[:-1]:
#         if value >= array[-1]:
#             break
#         low += 1
#     for value in array[-2::-1]:
#         if value < array[-1] or high < low:
#             break
#         high -= 1
#     array[low], array[high] = array[high], array[low]
#     return low
# partition(array, low, high)


# def select_pivot(sub_array):
#     if len(sub_array) % 2:
#         pivot_index = len(sub_array) // 2
#     else:
#         pivot_index = len(sub_array) // 2 - 1
#     return pivot_index


if __name__ == '__main__':
    # test = [21,14, 4, 1, 3, 9, 20, 25, 6, 21, 14]
    # test = [2, 8, 7, 1, 3, 5, 6, 4]
    test = [1, 2, 3, 4]
    print(quicksort(test, 0, len(test) - 1))
    # print(quicksort(test, 0, len(test) - 1))
