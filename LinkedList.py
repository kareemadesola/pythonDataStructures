# from collections.abc import Sequence


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

    def __repr__(self):
        return self.data


class LinkedList:
    def __init__(self, nodes=None):
        self.head = None
        if nodes is not None:
            current_node = Node(data=nodes.pop(0))
            self.head = nodes.pop()
            for elem in nodes:
                current_node.next = Node(data=elem)
                current_node = current_node.next

    def __repr__(self):
        # Hold a reference to the head node
        current_node = self.head

        # Hold a reference to the resulting linked list
        nodes = []
        while current_node is not None:
            nodes.append(current_node.data)
            current_node = current_node.next
        nodes.append("None")
        return " -> ".join(nodes)

    def __iter__(self):
        current_node = self.head
        while current_node is not None:
            yield current_node
            current_node = current_node.next

    def add_first(self, new_node):
        new_node.next = self.head
        self.head = new_node

    # My Implementation
    """
    1.Didn't give rise to the situation where list is empty
    2. Didn't make use of the __iter__ function implemented
    """

    def __len__(self) -> int:
        length = 0
        for _ in self:
            length += 1
        return length

    # def add_last(self, last_node):
    #     node = self.head
    #     while node.next is not None:
    #         node = node.next
    #     node.next = last_node

    def add_last(self, new_node):
        if self.head is None:
            self.head = new_node
            return
        for current_node in self:
            pass
        current_node.next = new_node

    def add_after(self, target_node_data, new_node):
        if self.head is None:
            raise Exception("List is empty")

        for current_node in self:
            if current_node.data == target_node_data:
                new_node.next = current_node.next
                current_node.next = new_node
                return

        raise Exception(f'Node with data {target_node_data} not found')

    # def add_before(self, target_node_data, new_node):
    #     if self.head is None:
    #         raise Exception("List is empty")
    #     if self.head.data == target_node_data:
    #         return self.add_first(new_node)
    #     for current_node in self:
    #         if current_node.next.data == target_node_data:
    #             new_node.next = current_node.next
    #             current_node.next = new_node
    #             return
    #     raise Exception(f"Node with data {target_node_data} not found")

    def add_before(self, target_node_data, new_node):
        if self.head is None:
            raise Exception("List is empty")

        if self.head.data == target_node_data:
            return self.add_first(new_node)

        prev_node = self.head
        for current_node in self:
            if current_node.data == target_node_data:
                prev_node.next = new_node
                new_node.next = current_node
                return
            prev_node = current_node

        raise Exception("Node with data '%s' not found" % target_node_data)

    def add_at_position(self, index, new_node):
        # if index >= len(self):
        #     self.add_last(new_node)
        # raise Exception("IndexError List index out of range")
        if index == 0:
            self.add_first(new_node)
        else:
            for current_index, current_node in enumerate(self):
                if current_index == index - 1:
                    new_node.next = current_node.next
                    current_node.next = new_node

    def remove_node(self, target_node_data):
        if self.head is None:
            raise Exception("List is empty")

        if self.head.data == target_node_data:
            self.head = self.head.next
            return

        prev_node = self.head
        for current_node in self:
            if current_node.data == target_node_data:
                prev_node.next = current_node.next
                return
            prev_node = current_node

        raise Exception(f"Node with data {target_node_data} not found")

    def __getitem__(self, index):
        if index < 0:
            index = len(self) + index
        if index >= len(self) or index < 0:
            raise Exception("IndexError: list index out of range")
        for current_index, current_node in enumerate(self):
            if current_index == index:
                return current_node

    def pop(self):
        for current_node in self:
            if current_node.next.next is None:
                last_node = current_node.next
                current_node.next = None
                return last_node

    # def __reversed__(self):
    # """This implementation was not correct"""
    #     for current_index, current_node in enumerate(self):
    #         self.add_at_position(current_index, self.pop())

    def __reversed__(self):
        current_node = self.head
        previous_node = None
        while current_node is not None:
            next_node = current_node.next
            current_node.next = previous_node
            previous_node = current_node
            current_node = next_node
        self.head = previous_node


if __name__ == '__main__':
    llist = LinkedList()
    print(llist)

    first_node = Node("a")
    llist.head = first_node
    print(llist)

    second_node = Node("b")
    third_node = Node("c")

    first_node.next = second_node
    second_node.next = third_node
    print(llist)

    for node in llist:
        print(node)

    llist.add_first(Node("d"))
    print(llist)

    llist.add_last(Node("e"))
    print(llist)
    llist.add_last(Node("f"))
    print(llist)
    llist.add_after("c", Node("g"))
    print(llist)
    llist.add_after("f", Node("h"))
    print(llist)
    # llist.add_after("y", Node("z"))
    # print(llist)
    llist.add_before("c", Node("i"))
    print(llist)
    llist.add_before("a", Node("j"))
    print(llist)
    llist.add_before("d", Node("q"))
    print(llist)
    llist.add_before("d", Node("k"))
    print(llist)
    llist.remove_node("c")
    print(llist)
    llist.remove_node("q")
    print(llist)
    # llist.remove_node("z")
    print(llist)
    # node_test = []
    # for i in llist:
    #     node_test.append(i)
    # print(node_test)
    # print(llist[-15])
    print(len(llist))
    llist.pop()
    # print(llist.pop())
    print(llist)
    llist.__reversed__()
    print(llist)
    print(llist.head)
    llist.add_first(Node("z"))
    print(llist)
    llist.__reversed__()
    print(llist)
    print(llist.head)
    llist.add_first(Node("y"))
    print(llist)
    llist.__reversed__()
    print(llist)
    print(llist.head)
