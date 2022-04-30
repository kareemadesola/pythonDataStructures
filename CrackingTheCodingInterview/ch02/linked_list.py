import random
import unittest
from copy import copy
from typing import Union, List, Optional


class LinkedListNode:
    def __init__(self, value: Union[int, str], next_node=None, prev_node=None):
        self.value = value
        self.next: Optional[LinkedListNode] = next_node
        self.prev: Optional[LinkedListNode] = prev_node

    def __str__(self):
        return str(self.value)


class LinkedList:
    def __init__(self, values=None):
        self.head: Optional[LinkedListNode] = None
        self.tail: Optional[LinkedListNode] = None
        if values:
            self.add_multiple(values)

    def __iter__(self):
        current_node = self.head
        while current_node is not None:
            yield current_node
            current_node = current_node.next

    def __str__(self):
        return " -> ".join(str(i) for i in self)

    def __len__(self):
        length = 0
        for _ in self:
            length += 1
        return length

    def values(self):
        return [i.value for i in self]

    def add(self, value: Union[int, str]):
        if self.head is None:
            self.head = self.tail = LinkedListNode(value)
        else:
            self.tail.next = LinkedListNode(value)
            self.tail = self.tail.next
        return self.tail

    def add_to_beginning(self, value: Union[int, str]):
        if not self.head:
            self.head = self.tail = LinkedListNode(value)
        else:
            self.head = LinkedListNode(value, self.head)
        return self.head

    def add_multiple(self, values: List[Union[int, str]]):
        for v in values:
            self.add(v)

    @classmethod
    def generate(cls, k: int, min_value: int, max_value: int):
        return cls(random.choices(range(min_value, max_value), k=k))


class DoublyLinkedList(LinkedList):
    def add(self, value: Union[int, str]):
        if not self.head:
            self.head = self.tail = LinkedListNode(value)
        else:
            self.tail.next = LinkedListNode(value, None, self.tail)
            self.tail = self.tail.next
        return self.tail


class Test(unittest.TestCase):
    ll = LinkedList([1, 2, 3, 4, 5])

    def test_add_to_beginning(self):
        ll = copy(self.ll)
        ll.add_to_beginning(5)
        self.assertTrue(ll.values(), [5, 1, 2, 3, 4, 5])

    def test_value(self):
        self.assertTrue(self.ll.values(), [1, 2, 3, 4])

    def test_len(self):
        self.assertTrue(len(self.ll), 5)

    def test_str(self):
        self.assertTrue(self.ll, "1 -> 2 -> 3 -> 4 -> 5")

    def test_generate(self):
        self.assertTrue(len(LinkedList.generate(10, 2, 8)), 10)
