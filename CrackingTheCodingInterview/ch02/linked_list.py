import abc
import random
import unittest
from copy import copy
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
        return " <--->>> ".join(str(i) for i in self)

    def add(self, value: Union[int, str]):
        if not self.head:
            self.head = self.tail = DoublyLinkedListNode(value)
        else:
            self.tail.next = DoublyLinkedListNode(value, None, self.tail)
            self.tail = self.tail.next
        return self.tail


class CircularSinglyLinkedList(SinglyLinkedList, LinkedListInterface):
    def __iter__(self):
        current_node = self.head
        while current_node.next != self.head:
            yield current_node
            current_node = current_node.next

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
        pass
        # if self.head is None:
        #     self.head = self.tail = SinglyLinkedListNode(value)




# class Circular
class Test(unittest.TestCase):
    ll = SinglyLinkedList([1, 2, 3, 4, 5])

    def test_singly_add(self):
        ll = copy(self.ll)
        ll.add(7)
        self.assertTrue(ll.values(), [5, 1, 2, 3, 4, 5, 7])
        print(ll)

    def test_doubly_add(self):
        ll = DoublyLinkedList([1, 2, 3, 10])
        ll.add(17)
        self.assertTrue(ll.values(), [1, 2, 3, 10, 17])
        print(ll)
        print(ll.head)

    def test_add_to_beginning(self):
        ll = copy(self.ll)
        ll.add_to_beginning(5)
        self.assertTrue(ll.values(), [5, 1, 2, 3, 4, 5])
        print(ll)

    def test_value(self):
        self.assertTrue(self.ll.values(), [1, 2, 3, 4])

    def test_len(self):
        self.assertTrue(len(self.ll), 5)

    def test_str(self):
        self.assertTrue(self.ll, "1 -> 2 -> 3 -> 4 -> 5")

    def test_generate(self):
        self.assertTrue(len(SinglyLinkedList.generate(10, 2, 8)), 10)


if __name__ == '__main__':
    print(issubclass(SinglyLinkedList, LinkedListInterface))
    print(issubclass(DoublyLinkedList, SinglyLinkedList))
    a = SinglyLinkedList()
    print(isinstance(a, LinkedListInterface))
