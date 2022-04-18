"""
Strengths:
Intuitive
Space efficient
Fast on a sorted list
Stable

Weaknesses:
Slow
"""


def insertion_sort(array):
    for i in range(1, len(array)):
        for j in range(i, 0, -1):
            if not array[j] < array[j - 1]:
                break
            else:
                array[j], array[j - 1] = array[j - 1], array[j]
    return array


if __name__ == "__main__":
    test = [8, 3, 2, 7, 9, 1, 4]
    print(insertion_sort(test))
