import unittest
from copy import deepcopy

from CrackingTheCodingInterview.ch02.linked_list import SinglyLinkedList, SinglyLinkedListNode


def delete_middle_node(node: SinglyLinkedListNode):
    node.value = node.next.value
    node.next = node.next.next


class Test(unittest.TestCase):
    sll = SinglyLinkedList.generate(3, 2, 10)
    middle_node = sll.add(4)
    sll.add_multiple([4, 5])

    def test_delete_middle_node(self):
        print(self.sll)
        delete_middle_node(self.middle_node)
        print(self.sll)
