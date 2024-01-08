class DoublyListNode:
    def __init__(self, val: int, next_node=None, prev_node=None):
        self.val = val
        self.next: DoublyListNode = next_node
        self.prev: DoublyListNode = prev_node


class DoublyLinkedList:
    def __init__(self, head=None):
        self.head: DoublyListNode = head
        self.size = 0 if not head else len(self)

    def __len__(self):
        """helper function to get length of linked list"""
        curr = self.head
        size = 0
        while curr:
            curr = curr.next
            size += 1
        return size

    def get(self, index: int) -> int:
        if index >= self.size:
            return -1
        curr = self.head
        get = 0
        while curr:
            if index == get:
                return curr.val
            curr = curr.next
            get += 1

    def addAtHead(self, val: int) -> None:
        new_node = DoublyListNode(val, self.head)
        if self.head:
            self.head.prev = new_node
        self.head = new_node
        self.size += 1

    def addAtTail(self, val: int) -> None:
        if not self.head:
            return self.addAtHead(val)
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = DoublyListNode(val, prev_node=curr)
        self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index == 0:
            return self.addAtHead(val)
        if index == self.size:
            return self.addAtTail(val)
        if index > self.size:
            return
        curr = self.head.next
        get = 1
        while curr:
            if index == get:
                new_node = DoublyListNode(val, curr, curr.prev)
                curr.prev.next = new_node
                curr.prev = new_node
                self.size += 1
                return
            curr = curr.next
            get += 1

    def deleteAtIndex(self, index: int) -> None:
        if not self.head:
            return
        if index >= self.size:
            return
        if index == 0:
            if self.size == 1:
                self.head = None
                self.size -= 1
                return

            self.head = self.head.next
            self.head.prev.next = None
            self.head.prev = None
            self.size -= 1
            return
        curr = self.head.next
        get = 1
        while curr:
            if index == get:
                if index == self.size - 1:
                    curr.prev.next = None
                    curr.prev = None
                else:
                    curr.prev.next = curr.next
                    curr.next.prev = curr.prev
                self.size -= 1
                return
            curr = curr.next
            get += 1
