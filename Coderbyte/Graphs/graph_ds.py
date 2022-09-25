from typing import List, Set

vertices = ['A', 'B', 'C', 'D', 'E']
vertices_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
edges = [['A', 'B'], ['A', 'D'], ['B', 'C'], ['C', 'D'], ['C', 'E'], ['D', 'E']]
adjacency_matrix = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 1, 0], ]
adjacency_list = [['B', 'D'], ['A', 'C'], ['B', 'D', 'E'], ['A', 'C', 'E'], ['C', 'D']]


def find_adjacent_nodes(node: str):
    """Given two nodes, find its neighbours"""
    # time O(e)
    # space O(1)
    res = []
    for x, y in edges:
        if node == x:
            res.append(y)
        elif node == y:
            res.append(x)
    return res


def is_connected(node_1: str, node_2: str):
    """Given two nodes, return True
    if there is a connected"""
    # time O(e)
    # space O(1)
    for x, y in edges:
        if x == node_1 and y == node_2 or x == node_2 and y == node_1:
            return True
    return False


def is_connected_any(node_1: str, node_2: str):
    """Given two nodes, return True
    if there is a connected"""
    # time O(e)
    # space O(1)
    # return any(node_1 in edge and node_2 in edge for edge in edges)
    return any(x == node_1 and y == node_2 or x == node_2 and y == node_1 for x, y in edges)


def find_adjacent_nodes_adjacency_matrix(node: str):
    # time O(v)
    # space O(1)
    """Given two nodes, find its neighbours"""
    target, res = vertices_index[node], []
    for idx, val in enumerate(adjacency_matrix[target]):
        if val:
            res.append(vertices[idx])
    return res


def test_find_adjacent_nodes_adjacency_matrix():
    assert ['D', 'B'].sort() == find_adjacent_nodes_adjacency_matrix('A').sort()


def is_connected_adjacency_matrix(node_1: str, node_2: str):
    # time O(1)
    # space O(1)
    """Given two nodes, return True
    if it is connected"""
    idx_1, idx_2 = vertices_index[node_1], vertices_index[node_2]
    return bool(adjacency_matrix[idx_1][idx_2])


def test_is_connected_adjacency_matrix():
    assert is_connected_adjacency_matrix('A', 'B')
    # assert is_connected_adjacency_matrix('B', 'E')
    assert is_connected_adjacency_matrix('E', 'D')


def find_adjacent_nodes_adjacency_list(node: str) -> List[str]:
    return adjacency_list[vertices_index[node]]


def test_find_adjacent_nodes_adjacency_list():
    assert ['D', 'B'].sort() == find_adjacent_nodes_adjacency_list('A').sort()


def is_connected_adjacency_list(node_1: str, node_2: str) -> bool:
    return node_2 in adjacency_list[vertices_index[node_1]]


def test_is_connected_adjacency_list():
    assert is_connected_adjacency_list('A', 'B')
    # assert is_connected_adjacency_matrix('B', 'E')
    assert is_connected_adjacency_list('E', 'D')


class Node:
    def __init__(self, val):
        self.val = val
        self.edges_set: Set[Node] = set()

    def connect(self, node):
        if node not in self.edges_set:
            self.edges_set.add(node)
            node.edges_set.add(self)

    def find_adjacent_nodes(self) -> List[str]:
        return [i.val for i in self.edges_set]

    def is_connected(self, node):
        return node in self.edges_set


class Graph:
    def __init__(self, nodes):
        self.nodes: Set[Node] = nodes

    def add_to_graph(self, new_node):
        if new_node not in self.nodes:
            self.nodes.add(new_node)


node_A = Node('A')
node_B = Node('B')
node_C = Node('C')
node_D = Node('D')
node_E = Node('E')
node_F = Node('F')

graph = Graph({node_A, node_B, node_C, node_D, node_E, node_F})

# print(str(node_A), node_B)
node_A.connect(node_B)
node_A.connect(node_D)
node_B.connect(node_C)
node_C.connect(node_E)
node_D.connect(node_E)


def test_get_adjacency_list_graph_class():
    assert ['B', 'D'] == sorted(node_A.find_adjacent_nodes())


def test_is_connected_adjacency_list_graph_class():
    assert not node_A.is_connected(node_C)
