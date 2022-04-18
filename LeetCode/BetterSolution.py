from typing import List


class BetterSolution:

    def divideArray(self, nums: List[int]) -> bool:
        if len(nums) % 2 != 0:
            return False
        for element in nums:
            if not nums.count(element) % 2 == 0:
                return False
        return True

    def maximumSubsequenceCount(self, text: str, pattern: str) -> int:
        pass

    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        count = 0
        max_ones = []
        for i in nums:
            if i != 1:
                max_ones.append(count)
                count = 0
                continue
            count += 1
        max_ones.append(count)
        return max(max_ones)


if __name__ == '__main__':
    test = BetterSolution()
    print(test.findMaxConsecutiveOnes([1, 1, 0, 1, 1, 1, ]))
