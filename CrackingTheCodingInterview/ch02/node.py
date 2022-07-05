from typing import Optional


class Node:
    def __init__(self, data: int):
        self.next_node: Optional[Node] = None
        self.data = data

    def append_to_tail(self, data: int):
        end = Node(data)
        n = self
        while n.next_node:
            n = n.next_node
        n.next_node = end

    def delete_node(self, data: int):
        if self.data == data:
            return self.next_node
        while self.next_node:
            if self.next_node.data == data:
                self.next_node = self.next_node.next_node
                return self
            self = self.next_node
        return self


if __name__ == '__main__':
    head = Node(4)

    head.append_to_tail(5)
    head.append_to_tail(2)
    b = head
    b.delete_node(5)
    print(b.next_node.data)
    print(head.next_node.data)
