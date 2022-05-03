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


if __name__ == '__main__':
    a = Node(4)
    b = Node(1)
    a.append_to_tail(5)
    a.append_to_tail(2)
    print(a.next_node.next_node.data)
    print(b.next_node.next_node.data)
