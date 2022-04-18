"""
selection sort
Time complexity for all cases O(n^2)
Strength
Space efficient

Weakness
Slow: 0(n^2) even if the array is sorted
Unstable
Implementation Trick
Get the index of smallest index

"""


def selection_sort(array):
    for i in range(len(array) - 1):
        smallest_index = i
        for j in range(i + 1, len(array)):
            if array[j] < array[smallest_index]:
                smallest_index = j
        array[smallest_index], array[i] = array[i], array[smallest_index]
    return array


if __name__ == '__main__':
    test = [8, 3, 2, 7, 9, 1, 4]
    print(selection_sort(test))
