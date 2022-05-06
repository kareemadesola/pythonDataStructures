import unittest

from CrackingTheCodingInterview.ch02.linked_list import SinglyLinkedList


def partition(sll: SinglyLinkedList, ptt: int):
    left_sll = SinglyLinkedList()
    right_sll = SinglyLinkedList()
    for i in sll:
        if i.value < ptt:
            left_sll.add(i)
        else:
            right_sll.add(i)
    left_sll.add_multiple(right_sll.values())
    return left_sll


def partition_head_tail(sll: SinglyLinkedList, ptt: int):
    current = sll.tail = sll.head
    current = current.next

    while current:
        next_node = current.next
        # current.next = None
        if current.value < ptt:
            current.next = sll.head
            sll.head = current
        else:
            sll.tail.next = current
            sll.tail = current
        current = next_node

    if sll.tail.next:
        sll.tail.next = None


class Test(unittest.TestCase):

    def test_partition(self):
        sll = SinglyLinkedList.generate(6, 1, 10, )
        print(sll)
        print("hello how are you")
        print(partition(sll, 5))

    def test_partition_head_tail(self):
        sll = SinglyLinkedList([1, 2, 3, 1, 4, 2])
        print(sll)
        partition_head_tail(sll, 2)
        print(sll)
