import pprint
from typing import List, Union, Optional


class Node:
    def __init__(self, value: int):
        self.value = value
        self.edges: List[Edge] = []
        self.visited = False


class Edge:
    def __init__(self, value: int, node_from: Node, node_to: Node):
        self.value = value
        self.node_from = node_from
        self.node_to = node_to


# You only need to change code with docs strings that have TODOs.
# Specifically: Graph.dfs_helper and Graph.bfs
# New methods have been added to associate node numbers with names
# Specifically: Graph.set_node_names
# and the methods ending in "_names" which will print names instead
# of node numbers

class Graph:
    def __init__(self, nodes=None, edges=None):
        self.nodes: List[Node] = nodes or []
        self.edges: List[Edge] = edges or []
        self.node_names = []
        self._node_map = {}

    def set_node_names(self, names):
        """The Nth name in names should correspond to node number N.
        Node numbers are 0 based (starting at 0)"""
        self.node_names = list(names)

    def insert_node(self, new_node_val:int)->Node:
        """Insert a new node with value new_node_val"""
        new_node = Node(new_node_val)
        self.nodes.append(new_node)
        self._node_map[new_node_val] = new_node
        return new_node

    def insert_edge(self, new_edge_val:int, node_from_val:int, node_to_val:int):
        """Insert a new edge, creating new nodes if necessary"""
        from_found:Optional[Node] = None
        to_found:Optional[Node] = None
        for node in self.nodes:
            if node.value == node_from_val:
                from_found = node
            if node.value == node_to_val:
                to_found = node
        if from_found is None:
            from_found = Node(node_from_val)
        if to_found is None:
            to_found = Node(node_to_val)
        new_edge = Edge(new_edge_val, from_found, to_found)
        from_found.edges.append(new_edge)
        to_found.edges.append(new_edge)
        self.edges.append(new_edge)







if __name__ == '__main__':
    graph = Graph()
    # You do not need to change anything below this line.
    # You only need to implement Graph.dfs_helper and Graph.bfs

    # pp = pprint.PrettyPrinter(indent=2)
    # print("Edge List")