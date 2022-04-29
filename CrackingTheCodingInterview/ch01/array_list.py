import ctypes
import unittest
from copy import copy


class ArrayList:
    def __init__(self, max_size=1):
        self._max_size = max_size
        self._size = 0
        self.A = self._make_array(max_size)

    def __len__(self) -> int:
        return self._size

    def __str__(self) -> str:
        if not len(self):
            return "[]"
        return ", ".join(str(i) for i in self.A[:len(self)])

    def __getitem__(self, index: int):
        if not -self._size <= index < self._size:
            raise IndexError(f"Index {index} out of range")
        if index >= 0:
            return self.A[index]
        if index < 0:
            return self.A[len(self) + index]

    def __contains__(self, element):
        return element in self.A[:self._size]

    def __setitem__(self, index, element):
        if not -self._size <= index < self._size:
            raise IndexError(f"List Assignment index {index} out of range")
        if index < 0:
            index += self._size
        self.A[index] = element

    def insert(self, index: int, element: int):
        if not -self._size <= index < self._size:
            self.append(element)
            self._size += 1
            return
        if self._size == self._max_size:
            self._resize(2 * self._max_size)
        if index < 0:
            index += self._size
        for i in reversed(range(index, self._size)):
            self.A[i + 1] = self.A[i]
        self.A[index] = element
        self._size += 1

    def append(self, element):
        if self._size == self._max_size:
            self._resize(2 * self._max_size)
        self.A[self._size] = element
        self._size += 1

    def pop(self, index=-1):
        if not -self._size <= index < self._size:
            raise IndexError(f"List Assignment Index {index} out of range")
        if index < 0:
            index += len(self)
        if index == self._size - 1:
            self._size -= 1
            return
        for i in range(index, self._size):
            self.A[index] = self.A[index + 1]
        self._size -= 1

    def remove(self, element):
        index = None
        for i in range(len(self)):
            if self.A[i] == element:
                index = i
                break
        if index:
            self.pop(index)

    def replace(self, index, element):
        self.__setitem__(index, element)

    def clean(self):
        self._size = 0

    def empty(self):
        return self._size == 0

    def _make_array(self, new_capacity):
        return (new_capacity * ctypes.py_object)()

    def _resize(self, new_capacity):
        B = self._make_array(new_capacity)

        # copy all element in A
        for i in range(self._size):
            B[i] = self.A[i]

        self.A = B
        self._max_size = new_capacity


class Test(unittest.TestCase):
    a = ArrayList(2)
    a.append(5)
    a.append(4)
    a.append(1)

    def test_append(self):
        self.assertEqual(str(self.a), '5, 4, 1')

    def test_length(self):
        self.assertEqual(len(self.a), 3)

    def test_get_item(self):
        self.assertEqual(self.a[0], 5)
        self.assertEqual(self.a[-2], 4)

    def test_contains(self):
        self.assertTrue(5 in self.a)
        self.assertFalse(2 in self.a)

    def test_set_item(self):
        a = copy(self.a)
        a[2] = 3
        self.assertTrue(a[2], 3)

    def test_replace(self):
        a = copy(self.a)
        a.replace(0, 100)
        self.assertTrue(a[0], 100)

    def test_pop(self):
        a = copy(self.a)
        self.assertEqual(len(a), 3)
        a.pop()
        self.assertEqual(len(a), 2)

    def test_remove(self):
        a = copy(self.a)
        self.assertEqual(len(a), 3)
        a.remove(3)
        self.assertEqual(len(a), 3)
        a.remove(5)
        # self.assertEqual(len(a), 2)


if __name__ == '__main__':
    unittest.main()
