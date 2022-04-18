#!/bin/python3

import os


#
# Complete the 'lonelyinteger' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY a as parameter.
#

def lonely_integer(a):
    n = len(a)
    index = 0
    while n != 0:
        if a[index] in a[index+1:]:
            test= a[index]
            a.pop(test)
            a.pop(test)
        else:
            a[index + 1:].remove(a[index])
            index += 1
        n = len(a)

    # Write your code here
    # for index in range(len(a)):
    #     if not a[index] in a[index+1:]:
    #         return a[index]


if __name__ == '__main__':
    # fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    a = list(map(int, input().rstrip().split()))

    result = lonely_integer(a)
    print(result)
    # fptr.write(str(result) + '\n')

    # fptr.close()
