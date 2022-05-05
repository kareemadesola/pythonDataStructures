import unittest
from copy import deepcopy
from typing import Optional

from linked_list import SinglyLinkedList


def remove_dups_brute_force_with_set(sll: SinglyLinkedList):
    seen = set()
    for i in sll:
        if i.value not in seen:
            seen.add(i.value)
    sll = SinglyLinkedList(seen)
    return sll


def remove_dups_better_with_set(sll: SinglyLinkedList) -> SinglyLinkedList:
    current = sll.head
    previous: Optional[SinglyLinkedList] = None
    seen = set()

    while current:
        if current.value in seen:
            previous.next = current.next
        else:
            seen.add(current.value)
            previous = current
        current = current.next
    sll.tail = previous
    return sll


def remove_dups_no_extra_space(sll: SinglyLinkedList) -> SinglyLinkedList:
    current = runner = sll.head
    while current:
        runner = current
        while runner.next:
            if runner.next.value == current.value:
                runner.next = runner.next.next
            else:
                runner = runner.next
        current = current.next
    sll.tail = runner
    return sll


class Test(unittest.TestCase):
    sll = SinglyLinkedList.generate(12, 1, 5)

    def test_remove_dups_brute_force_with_set(self):
        sll = deepcopy(self.sll)
        sll = remove_dups_brute_force_with_set(sll)
        self.assertLessEqual(len(sll), 4)

    def test_remove_dups_better_with_set(self):
        sll = deepcopy(self.sll)
        remove_dups_better_with_set(sll)
        self.assertLessEqual(len(sll), 4)

    def test_remove_dups_no_extra_space(self):
        sll = deepcopy(self.sll)
        remove_dups_no_extra_space(sll)
        self.assertLessEqual(len(sll), 4)
