import unittest
from copy import deepcopy

from CrackingTheCodingInterview.ch02.linked_list import SinglyLinkedList


def return_kth_to_last_using_len(sll: SinglyLinkedList, k: int):
    current = sll.head
    index = len(sll) - k
    if index < 0:
        return None
    for _ in range(index):
        current = current.next
    return current.value


def return_kth_to_last_using_runner(sll: SinglyLinkedList, k: int):
    current = runner = sll.head
    for _ in range(k):
        if not runner:
            return None
        runner = runner.next
    while runner:
        current = current.next
        runner = runner.next
    return current.value


def return_kth_last_recursive(ll, k):
    head = ll.head
    counter = 0

    def helper(head, k):
        nonlocal counter
        if not head:
            return None
        helper_node = helper(head.next, k)
        counter = counter + 1
        if counter == k:
            return head
        return helper_node

    return helper(head, k)


def return_kth_last_recursive_index(sll: SinglyLinkedList, k: int):
    head = sll.head

    def helper(head, k):
        if not head:
            return 0
        index = helper(head.next, k) + 1
        if index == k:
            print(f"{k}th to the last node is {head.value}")
        return index

    helper(head, k)


class Test(unittest.TestCase):
    sll = SinglyLinkedList([12, 13, 14, 2, 5, 6, 9])

    def test_return_kth_to_last_using_len(self):
        sll = deepcopy(self.sll)
        print(sll)
        self.assertEqual(return_kth_to_last_using_len(sll, 2), 6)

    def test_return_kth_to_last_using_runner(self):
        sll = deepcopy(self.sll)
        print(sll)
        self.assertEqual(return_kth_to_last_using_runner(sll, 2), 6)

    def test_return_kth_to_last_using_runner_edge(self):
        sll = deepcopy(self.sll)
        print(sll)
        self.assertEqual(return_kth_to_last_using_runner(sll, 7), 12)

    def test_return_kth_to_last_recursive(self):
        sll = deepcopy(self.sll)
        print(sll)
        self.assertEqual(return_kth_last_recursive(sll, 7).value, 12)

    def test_return_kth_to_last_recursive_index(self):
        sll = SinglyLinkedList([13, 15, 17, 20])
        print(sll)
        # self.assertEqual(return_kth_last_recursive_index(sll, 7).value, 12)
        return_kth_last_recursive_index(sll, 2)
