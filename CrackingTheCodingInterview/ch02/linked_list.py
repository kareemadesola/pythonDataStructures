from typing import Union, List


class LinkedListNode:
    def __init__(self, value: Union[int, str], next_node=None, prev_node=None):
        pass

    def __str__(self):
        pass


class LinkedList:
    def __init__(self, values=None):
        pass

    def __iter__(self):
        pass

    def __str__(self):
        pass

    def __len__(self):
        pass

    def values(self):
        pass

    def add(self, value: Union[int, str]):
        pass

    def add_to_beginning(self, value: Union[int, str]):
        pass

    def add_multiple(self, values: List[Union[int, str]]):
        pass

    @classmethod
    def generate(cls, k: int, min_value: int, max_value: int):
        pass


class DoublyLinkedList(LinkedList):
    def add(self, value: Union[int, str]):
        pass
