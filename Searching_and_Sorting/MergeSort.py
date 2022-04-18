def abstract_in_place_merge(array):
    # assumes it initial array has sorted halves
    # eg [4, 5, 11, 12, 12, 1, 7, 9, 11, 13]
    # [4,5,11,12,12] and [1,7,9,11,13] are sorted
    auxiliary_array = array.copy()
    index = 0
    low = 0
    high = 0
    for first_index, first_sub_element in enumerate(auxiliary_array[:len(array) // 2]):
        for second_index, second_sub_element in enumerate(auxiliary_array[len(array) // 2:]):
            if first_sub_element <= second_sub_element:
                array[index] = first_sub_element
                low += 1
                index += 1
                break
            elif second_index >= high:
                array[index] = second_sub_element
                high += 1
                index += 1
            if high == len(array) // 2 or low == len(array) // 2:
                if low > high:
                    array[index] = auxiliary_array[-1]
                elif high > low:
                    array[index] = auxiliary_array[len(array) // 2 - 1]
    return array


if __name__ == '__main__':
    print(abstract_in_place_merge([4, 5, 1, 7, ]))
    print(abstract_in_place_merge([4, 5, 11, 12, 1, 7, 9, 11]))
    print(abstract_in_place_merge([4, 5, 11, 12, 12, 1, 7, 9, 11, 13]))
