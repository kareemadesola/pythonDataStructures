from typing import List, Union


class DVD:
    def __init__(self, name, release_year, director):
        self.name = name
        self.release_year = release_year
        self.director = director

    def __repr__(self):
        return f"""{self.name}, directed by {self.director}, released in {self.release_year}"""


class Solution:
    def duplicateZeros(self, arr: List[int]) -> list[int]:
        """
        Do not return anything, modify arr in-place instead.
        """
        for i in range(len(arr)):
            if arr[i] == 0:
                arr.insert(i, 0)
                arr.pop()
                i += 2
        return arr


if __name__ == '__main__':
    a = Solution()
    print(a.duplicateZeros([1, 0, 2, 3, 0, 4, 5, 0]))
    dvdCollection: List[Union[DVD, None]] = [None] * 15

    incrediblesDVD = DVD("The Incredibles", 2004, "Brad Bird")
    findingDoryDVD = DVD("Finding Dory", 2016, "Andrew Stanton")
    lionKingDVD = DVD("The Lion King", 2019, "Jon Favreau")
    starWarsDVD = DVD("Star Wars", 1977, "George Lucas")
    print(lionKingDVD)

    dvdCollection[3] = incrediblesDVD
    dvdCollection[9] = findingDoryDVD
    dvdCollection[2] = lionKingDVD
    dvdCollection[3] = starWarsDVD
    print(dvdCollection)

    """Writing items into an Array with a Loop"""
    # squareNumbers: List[Union[int, None]] = [i ** 2 for i in range(1, 11)]
    # for i in squareNumbers:
    #     print(i)
    # print(squareNumbers)

    """Finding capacity of array"""
    print(len(dvdCollection))

    """Finding  the length of dvdCollection"""
    length = 0
    for dvd in dvdCollection:
        if dvd is not None:
            length += 1

    print(length)
