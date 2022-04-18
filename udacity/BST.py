class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BST:
    def __init__(self, root_value):
        self.root = Node(root_value)

    def insert(self, new_val):
        self.insert_helper(self.root, new_val)

    def insert_helper(self, current, new_val):
        if new_val > current.value:
            if current.right is None:
                current.right = Node(new_val)
                return
            return self.insert_helper(current.right, new_val)
        if current.left is None:
            current.left = Node(new_val)
            return
        return self.insert_helper(current.left, new_val)

    def search(self, find_val):
        self.search_helper(self.root, find_val)

    def search_helper(self, current: Node, find_val: int):
        if current:
            if current.value == find_val:
                return True
            elif find_val > current.value:
                return self.search_helper(current.right, find_val)
            else:
                return self.search_helper(current.left, find_val)

        return False


if __name__ == '__main__':
    # Set up tree
    tree = BST(4)

    # Insert elements
    tree.insert(2)
    tree.insert(1)
    tree.insert(3)
    tree.insert(5)

    # Check search
    # Should be True
    print(tree.search(4))
    # Should be False
    print(tree.search(6))
