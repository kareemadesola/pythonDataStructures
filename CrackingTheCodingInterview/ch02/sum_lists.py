import unittest

from CrackingTheCodingInterview.ch02.linked_list import SinglyLinkedList


def sum_list_brute_force(sll1: SinglyLinkedList, sll2: SinglyLinkedList):
    return SinglyLinkedList(
        list(map(int,
                 str(int(''.join(str(i) for i in sll1.values())[::-1]) +
                     int(''.join(str(i) for i in sll2.values())[::-1]))[::-1])))


class Test(unittest.TestCase):
    sll1 = SinglyLinkedList([7, 1, 6])
    sll2 = SinglyLinkedList([5, 9, 2])

    def test_sum_list_brute_force(self):
        print(sum_list_brute_force(self.sll1, self.sll2))
        # self.assertEqual(sum_list_brute_force(self.sll1, self.sll2),
        # SinglyLinkedList([2,1,9]))
