from typing import List


class DailyCodingProblem:
    def two_numbers_add_up_to_k(self, arr: List[int], k: int):
        for i in range(len(arr) - 1):
            for j in range(i + 1, len(arr)):
                if arr[i] + arr[j] == k:
                    return True
        return False

    def two_numbers_add_up_to_k_tayo(self, arr: List[int], k: int):
        hashmap = {}
        for i, x in enumerate(arr):
            if k - x in hashmap:
                return True
            else:
                hashmap[x] = i
        return False

    def two_numbers_add_up_to_k_tayo_modified(self, arr: List[int], k: int):
        for i in arr:
            if k - i in arr:
                return True
        return False



if __name__ == '__main__':
    test = DailyCodingProblem()
    # print(test.two_numbers_add_up_to_k([10, 15, 3, 7], 17))
    print(test.two_numbers_add_up_to_k_tayo_modified([10, 15, 3, 7], 17))
    # print(test.two_numbers_add_up_to_k([10, 15, 3, 8], 17))
    print(test.two_numbers_add_up_to_k_tayo_modified([10, 15, 3, 8], 17))
