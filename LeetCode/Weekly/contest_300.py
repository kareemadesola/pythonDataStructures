import unittest


def decode_message(key: str, message: str) -> str:
    set_ = set()
    dict_ = {}
    alphabet = 97
    for i in key:
        if i not in set_ and i != " ":
            dict_[i] = chr(alphabet)
            set_.add(i)
            alphabet += 1
    dict_[" "] = " "
    return ''.join(dict_[i] for i in message)


class Test(unittest.TestCase):
    def test_minimum_numbers(self):
        self.assertEqual("this is a secret",
                         decode_message(key="the quick brown fox jumps over the lazy dog", message="vkbs bs t suepuv"))
