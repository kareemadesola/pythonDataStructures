class Node:
    def __init__(self, val: int, next_node=None):
        self.val = val
        self.next: Node = next_node


class MyLinkedList:

    def __init__(self, head=None):
        self.head: Node = head

    def __len__(self):
        count = 0
        ll = self.head
        while ll:
            ll = ll.next
            count += 1
        return count

    def get(self, index: int) -> int:
        if not 0 <= index < len(self):
            return -1
        ll = self.head
        for _ in range(index):
            ll = ll.next
        return ll.val

    def addAtHead(self, val: int) -> None:
        self.head = Node(val, self.head)

    def addAtTail(self, val: int) -> None:
        if not self.head:
            self.addAtHead(val)
            return
        ll = self.head
        for _ in range(len(self) - 1):
            ll = ll.next
        ll.next = Node(val)

    def addAtIndex(self, index: int, val: int) -> None:
        if not 0 <= index <= len(self):
            return
        if index == 0:
            return self.addAtHead(val)
        if index == len(self):
            return self.addAtTail(val)
        prev = self.head
        for _ in range(index - 1):
            prev = prev.next
        prev.next = Node(val, prev.next)

    def deleteAtIndex(self, index: int) -> None:
        if not 0 <= index < len(self):
            return
        if index == 0:
            self.head = self.head.next
            return
        prev = self.head
        for _ in range(index - 1):
            prev = prev.next
        prev.next = prev.next.next

# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
