import abc
import random
import unittest
from copy import deepcopy
from typing import Union, List, Optional


class SinglyLinkedListNode:
    def __init__(self, value: Union[int, str], next_node=None):
        self.value = value
        self.next: Optional[SinglyLinkedListNode] = next_node

    def __str__(self):
        return str(self.value)


class DoublyLinkedListNode(SinglyLinkedListNode):
    def __init__(self, value: Union[int, str], next_node=None, prev_node=None):
        super().__init__(value, next_node)
        self.prev: Optional[DoublyLinkedListNode] = prev_node


class LinkedListInterface(metaclass=abc.ABCMeta):
    """Person interface built from PersonMeta metaclass."""

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, '__init__') and \
               callable(subclass.__init__) and \
               hasattr(subclass, '__iter__') and \
               callable(subclass.__iter__) and \
               hasattr(subclass, '__str__') and \
               callable(subclass.__str__) and \
               hasattr(subclass, '__len__') and \
               callable(subclass.__len__) and \
               hasattr(subclass, 'values') and \
               callable(subclass.values) and \
               hasattr(subclass, 'add') and \
               callable(subclass.add) and \
               hasattr(subclass, 'add_to_beginning') and \
               callable(subclass.add_to_beginning) and \
               hasattr(subclass, 'add_multiple') and \
               callable(subclass.add_multiple) or \
               NotImplemented

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def values(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add(self, value: Union[int, str]):
        raise NotImplementedError

    @abc.abstractmethod
    def add_to_beginning(self, value: Union[int, str]):
        raise NotImplementedError

    @abc.abstractmethod
    def add_multiple(self, values: List[Union[int, str]]):
        raise NotImplementedError


class SinglyLinkedList(LinkedListInterface):
    def __init__(self, values=None):
        self.head: Optional[SinglyLinkedListNode] = None
        self.tail: Optional[SinglyLinkedListNode] = None
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
            self.head = self.tail = SinglyLinkedListNode(value)
        else:
            self.tail.next = SinglyLinkedListNode(value)
            self.tail = self.tail.next
        return self.tail

    def add_to_beginning(self, value: Union[int, str]):
        if not self.head:
            self.head = self.tail = SinglyLinkedListNode(value)
        else:
            self.head = SinglyLinkedListNode(value, self.head)
        return self.head

    def add_multiple(self, values: List[Union[int, str]]):
        for v in values:
            self.add(v)

    @classmethod
    def generate(cls, k: int, min_value: int, max_value: int):
        return cls(random.choices(range(min_value, max_value), k=k))


class DoublyLinkedList(SinglyLinkedList):
    def __str__(self):
        return " <-->> ".join(str(i) for i in self)

    def add(self, value: Union[int, str]):
        if not self.head:
            self.head = self.tail = DoublyLinkedListNode(value)
        else:
            self.tail.next = DoublyLinkedListNode(value, None, self.tail)
            self.tail = self.tail.next
        return self.tail

    def add_to_beginning(self, value: Union[int, str]):
        if not self.head:
            self.head = self.tail = DoublyLinkedListNode(value)
        else:
            new_node = DoublyLinkedListNode(value, self.head, None)
            self.head.prev = new_node
            self.head = new_node


class CircularSinglyLinkedList(SinglyLinkedList, LinkedListInterface):
    def __iter__(self):
        current_node = self.head
        while current_node:
            yield current_node
            if current_node.next != self.head:
                current_node = current_node.next
            else:
                break

    def add(self, value: Union[int, str]):
        if self.head is None:
            self.head = self.tail = SinglyLinkedListNode(value)
        else:
            new_node = SinglyLinkedListNode(value)
            new_node.next = self.tail.next
            self.tail.next = new_node
            self.tail = new_node
        return self.tail

    def add_to_beginning(self, value: Union[int, str]):
        if self.head is None:
            self.head = self.tail = SinglyLinkedListNode(value)
        else:
            new_node = SinglyLinkedListNode(value, self.head)
            self.head = new_node
            self.tail.next = new_node
        return self.head


class Test(unittest.TestCase):
    init_values = [1, 2, 3, 4, 5]
    sll = SinglyLinkedList(init_values)
    c_sll = CircularSinglyLinkedList(init_values)
    dll = DoublyLinkedList(init_values)
    """Test Circular Linked List"""

    def test_c_sll_add(self):
        c_sll = deepcopy(self.c_sll)
        c_sll.add(12)
        self.assertEqual(c_sll.values(), [1, 2, 3, 4, 5, 12])

    def test_c_sll_add_to_beginning_when_empty(self):
        c_sll = CircularSinglyLinkedList()
        c_sll.add_to_beginning(5)
        self.assertEqual(c_sll.values(), [5])

    def test_c_sll_add_to_beginning_when_not_empty(self):
        c_sll = deepcopy(self.c_sll)
        c_sll.add_to_beginning(5)
        self.assertEqual(c_sll.values(), [5, 1, 2, 3, 4, 5])

    def test_c_sll_len(self):
        c_sll = deepcopy(self.c_sll)
        self.assertEqual(len(c_sll), 5)

    """Test for Doubly Linked List"""

    def test_doubly_add(self):
        dll = deepcopy(self.dll)
        dll.add(17)
        self.assertEqual(dll.values(), [1, 2, 3, 4, 5, 17])

    def test_doubly_prev(self):
        dll = deepcopy(self.dll)
        dll.add_to_beginning(17)
        while dll.tail.prev:
            print(dll.tail.prev)
            dll.tail.prev = dll.tail.prev.prev
        print(dll.tail.prev)

    def test_doubly_add_to_beginning_when_empty(self):
        dll = DoublyLinkedList()
        dll.add_to_beginning(12)
        self.assertEqual(dll.values(), [12])

    def test_doubly_add_to_beginning_when_not_empty(self):
        dll = deepcopy(self.dll)
        dll.add_to_beginning(12)
        self.assertEqual(dll.values(), [12, 1, 2, 3, 4, 5])

    def test_doubly_generate(self):
        self.assertEqual(len(DoublyLinkedList.generate(10, 2, 8)), 10)

    """Test for Singly Linked List"""

    def test_singly_add(self):
        ll = deepcopy(self.sll)
        ll.add(7)
        self.assertEqual(ll.values(), [1, 2, 3, 4, 5, 7])

    def test_singly_add_to_beginning(self):
        ll = deepcopy(self.sll)
        ll.add_to_beginning(5)
        self.assertEqual(ll.values(), [5, 1, 2, 3, 4, 5])

    def test_sll_value(self):
        self.assertEqual(self.sll.values(), [1, 2, 3, 4, 5])

    def test_sll_len(self):
        self.assertEqual(len(self.sll), 5)

    def test_sll_str(self):
        self.assertEqual(str(self.sll), "1 -> 2 -> 3 -> 4 -> 5")

    def test_generate(self):
        self.assertEqual(len(SinglyLinkedList.generate(10, 2, 8)), 10)
