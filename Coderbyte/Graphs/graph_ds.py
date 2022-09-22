vertices = ['A', 'B', 'C', 'D', 'E']
edges = [['A', 'B'], ['A', 'D'], ['B', 'C'], ['C', 'D'], ['C', 'E'], ['D', 'E']]


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
