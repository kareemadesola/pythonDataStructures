def merge_sort(array):
    n = len(array)
    if n < 2:
        return
    middle_index = n // 2

    first_subarray = array[:middle_index]
    second_subarray = array[middle_index:]

    merge_sort(first_subarray)
    merge_sort(second_subarray)

    i = j = 0
    while i + j < len(array):
        if j == len(second_subarray) or \
                (i < len(first_subarray) and first_subarray[i] < second_subarray[j]):
            array[i + j] = first_subarray[i]
            i += 1
        else:
            array[i + j] = second_subarray[j]
            j += 1


if __name__ == "__main__":
    test = [1, 3, 6, 7, 0, 1, 4]
    print(merge_sort(test))
    print(test)
