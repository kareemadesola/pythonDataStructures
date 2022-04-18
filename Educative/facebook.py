# def move_zeros_to_left(array):
#     for i in range(len(array)):
#         if array[i] == 0:
#             for j in range(i, 0, -1):
#                 array[j], array[j - 1] = array[j - 1], array[j]
#                 if array[j] == 0:
#                     break
#     return array
#
def move_zeros_to_left(array):
    if len(array) < 2:
        return
    write_index = len(array) - 1
    for read_index in range(len(array) - 1, -1, -1):
        if array[read_index] != 0:
            array[write_index] = array[read_index]
            write_index -= 1

    for i in range(write_index + 1):
        array[i] = 0

    return array


if __name__ == "__main__":
    print(move_zeros_to_left([1, 10, 20, 0, 59, 63, 0, 88, 0]))
    print(move_zeros_to_left([1, 10, 20, 0, 59, 63, 0, 88, 0, 1, 0, 0]))
