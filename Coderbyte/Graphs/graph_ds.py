vertices = ['A', 'B', 'C', 'D', 'E']
vertices_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
edges = [['A', 'B'], ['A', 'D'], ['B', 'C'], ['C', 'D'], ['C', 'E'], ['D', 'E']]
adjacency_matrix = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 1, 0], ]


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
    # time O(e)
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
