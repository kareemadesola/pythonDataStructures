def kareem_sort(array):
    for i in range(len(array) - 1):
        print(i)
        for j in range(i + 1, len(array)):
            if array[i] > array[j]:
                array[i], array[j] = array[j], array[i]
            print(array)
    return array


def bubble_sort(array):
    for i in range(len(array) - 1):
        print(i)
        for j in range(len(array)-1-i):
            if array[j+1] < array[j]:
                array[j+1], array[j] = array[j], array[j+1]
            print(array)
    return array


def gcd(x, y):
    if x == 0 or y == 0:
        return max(x, y)
    # greater_number = max(x,y)
    # lower_number = min(x,y)

    return gcd(y, x % y)


if __name__ == '__main__':
    # print(kareem_sort([4, 3, 2, 1]))
    bubble_sort([6,5,4, 3, 2, 1])
    # print(gcd(2, 3))
    # print(gcd(20, 10))
