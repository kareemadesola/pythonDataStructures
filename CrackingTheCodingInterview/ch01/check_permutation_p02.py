import time
import unittest
from collections import defaultdict


def check_permutation_hash_table(str1, str2):
    if len(str1) > len(str2):
        return False
    str2_dict = {}
    # Put element in str2 into dictionary
    for i in str2:
        if i not in str2_dict:
            str2_dict[i] = 1
        else:
            str2_dict[i] += 1

    for i in str1:
        if i not in str2_dict:
            return False
        else:
            str2_dict[i] -= 1
            if str2_dict[i] < 0:
                return False
    return all(value == 0 for value in str2_dict.values())


def check_permutation_sorting(str1: str, str2: str):
    if len(str1) != len(str2):
        return False
    str1, str2 = ''.join(sorted(str1)), ''.join(sorted(str2))
    return str1 == str2


class Test(unittest.TestCase):
    # str1, str2, is_permutation
    test_cases = (
        ("dog", "god", True),
        ("abcd", "bacd", True),
        ("3563476", "7334566", True),
        ("wef34f", "wffe34", True),
        ("dogx", "godz", False),
        ("abcd", "d2cba", False),
        ("2354", "1234", False),
        ("dcw4f", "dcw5f", False),
        ("DOG", "dog", False),
        ("dog ", "dog", False),
        ("aaab", "bbba", False),
    )

    testable_functions = [
        # check_permutation_by_sort,
        # check_permutation_by_count,
        # check_permutation_pythonic,
        check_permutation_hash_table,
        check_permutation_sorting
    ]

    def test_cp(self):
        num_runs = 1000
        function_runtimes = defaultdict(float)

        # true check
        # for _ in range(num_runs):
        #     for check_permutation in self.testable_functions:
        #         for str1, str2, expected in self.test_cases:
        #             start = time.perf_counter()
        #             assert (check_permutation(str1, str2) == expected), \
        #                 f"{check_permutation.__name__} failed for value: {str1} {str2}"
        #             function_runtimes[check_permutation.__name__] += \
        #                 (time.perf_counter() - start) * 1000

        for _ in range(num_runs):
            for str1, str2, expected in self.test_cases:
                for check_permutation in self.testable_functions:
                    start = time.perf_counter()
                    assert (check_permutation(str1, str2) == expected), \
                        f"{check_permutation.__name__} failed for value: {str1} {str2}"
                    function_runtimes[check_permutation.__name__] += \
                        (time.perf_counter() - start) * 1000

        print(f"\n{num_runs} runs")
        for function_name, runtime in function_runtimes.items():
            print(f"{function_name}: {runtime:.1f}ms")


if __name__ == "__main__":
    unittest.main()
