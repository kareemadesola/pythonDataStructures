from typing import List, Union


class Node:
    def __init__(self, value):
        self.value = value
        self.edges = []


class Edge:
    def __init__(self, value: int, node_from: Node, node_to: Node):
        self.value = value
        self.node_from = node_from
        self.node_to = node_to


class Graph:
    def __init__(self, nodes: List[Node] = None,
                 edges: List[Edge] = None):
        if edges is None:
            edges = []
        if nodes is None:
            nodes = []
        self.nodes = nodes
        self.edges = edges

    def insert_node(self, new_node_val: int):
        new_node = Node(new_node_val)
        self.nodes.append(new_node)

    def insert_edge(self, new_edge_val, node_from_val, node_to_val):
        from_found: Union[Node | None] = None
        to_found: Union[Node | None] = None
        for node in self.nodes:
            if node_from_val == node.value:
                from_found = node
            if node_to_val == node.value:
                to_found = node
        if from_found is None:
            from_found = Node(node_from_val)
            self.nodes.append(from_found)
        if to_found is None:
            to_found = Node(node_to_val)
            self.nodes.append(to_found)
        new_edge = Edge(new_edge_val, from_found, to_found)
        from_found.edges.append(new_edge)
        to_found.edges.append(new_edge)
        self.edges.append(new_edge)

    def get_edge_list(self) -> List[tuple]:
        """Don't return a list of edge objects!
        Return a list of triples that looks like this:
        (Edge Value, From Node Value, To Node Value)"""
        return [(i.value, i.node_from.value, i.node_to.value)
                for i in self.edges]

    def get_adjacency_list(self):
        """Don't return any Node or Edge objects!
        You'll return a list of lists.
        The indices of the outer list represent
        "from" nodes.
        Each section in the list will store a list
        of tuples that looks like this:
        (To Node, Edge Value)"""
        max_num_nodes = self.get_node_max_value() + 1
        adjacent_list: List[List[tuple]] = [None] * max_num_nodes
        for i in self.edges:
            if adjacent_list[i.node_from.value]:
                adjacent_list[i.node_from.value].\
                    append((i.node_to.value, i.value,))
            else:
                adjacent_list[i.node_from.value] = \
                    [(i.node_to.value, i.value,)]
        return adjacent_list

    def get_adjacency_matrix(self):
        """Return a matrix, or 2D List.
        Row numbers represent from nodes,
        column numbers represent to nodes.
        Store the edge values in each spot,
        and a 0 if no edge exists"""
        max_num_nodes = self.get_node_max_value() + 1
        adjacency_matrix = [[0] * max_num_nodes for _ in range(max_num_nodes)]
        for i in self.edges:
            adjacency_matrix[i.node_from.value][i.node_to.value] =\
                i.value
        return adjacency_matrix

    def get_node_max_value(self):
        return max(max((i.node_from.value, i.node_to.value)
                       for i in self.edges))


if __name__ == '__main__':
    graph = Graph()
    graph.insert_edge(100, 1, 2)
    graph.insert_edge(101, 1, 3)
    graph.insert_edge(102, 1, 4)
    graph.insert_edge(103, 3, 4)
    # Should be [(100, 1, 2), (101, 1, 3), (102, 1, 4), (103, 3, 4)]
    print(graph.get_edge_list())
    # Should be
    # [None, [(2,100), (3, 101), (4, 102)], None, [(4, 103)], None]
    print(graph.get_adjacency_list())
    # Should be [[0, 0, 0, 0, 0], [0, 0, 100, 101, 102],
    # [0, 0, 0, 0, 0], [0, 0, 0, 0, 103], [0, 0, 0, 0, 0]]
    print(graph.get_adjacency_matrix())
