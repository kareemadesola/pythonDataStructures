import pprint
from typing import List, Optional, Tuple


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

    def insert_node(self, new_node_val: int) -> Node:
        """Insert a new node with value new_node_val"""
        new_node = Node(new_node_val)
        self.nodes.append(new_node)
        self._node_map[new_node_val] = new_node
        return new_node

    def insert_edge(self, new_edge_val: int, node_from_val: int,
                    node_to_val: int):
        """Insert a new edge, creating new nodes if necessary"""
        from_found: Optional[Node] = None
        to_found: Optional[Node] = None
        for node in self.nodes:
            if node.value == node_from_val:
                from_found = node
            if node.value == node_to_val:
                to_found = node
        if from_found is None:
            from_found = self.insert_node(node_from_val)
        if to_found is None:
            to_found = self.insert_node(node_to_val)
        new_edge = Edge(new_edge_val, from_found, to_found)
        from_found.edges.append(new_edge)
        to_found.edges.append(new_edge)
        self.edges.append(new_edge)

    def get_edge_list(self):
        """Return a list of triples that looks like this:
        (Edge Value, From Node, To Node)"""
        return [(e.value, e.node_from.value, e.node_to.value
                 ) for e in self.edges]

    def get_edge_list_names(self):
        """Return a list of triples that looks like this:
        (Edge Value, From Node Name, To Node Name)"""
        return [(edge.value,
                 self.node_names[edge.node_from.value],
                 self.node_names[edge.node_to.value])
                for edge in self.edges]

    def get_adjacency_list(self):
        """Return a list of lists.
        The indices of the outer list represent "from" nodes.
        Each section in the list will store a list
        of tuples that looks like this:
        (To Node, Edge Value)"""
        max_num_nodes = self.find_max_index()
        adjacency_list: List[Optional[List]] = [None] * max_num_nodes
        for edge in self.edges:
            if adjacency_list[edge.node_from.value]:
                adjacency_list[edge.node_from.value] \
                    .append((edge.node_to.value, edge.value))
            else:
                adjacency_list[edge.node_from.value] = \
                    [(edge.node_to.value, edge.value)]
        return adjacency_list

    def get_adjacency_list_names(self):
        """Each section in the list will store a list
        of tuples that looks like this:
        (To Node Name, Edge Value).
        Node names should come from the names set
        with set_node_names."""
        max_num_nodes = self.find_max_index()
        adjacency_list_names: List[Optional[List[Tuple]]] = \
            [None for _ in range(max_num_nodes)]
        for edge in self.edges:
            if adjacency_list_names[edge.node_from.value]:
                adjacency_list_names[edge.node_from.value] \
                    .append(
                    (self.node_names[edge.node_to.value], edge.value))
            else:
                adjacency_list_names[edge.node_from.value] = \
                    [(self.node_names[edge.node_to.value], edge.value)]
        return adjacency_list_names

    def get_adjacency_matrix(self):
        """Return a matrix, or 2D list.
        Row numbers represent from nodes,
        column numbers represent to nodes.
        Store the edge values in each spot,
        and a 0 if no edge exists."""
        max_index = self.find_max_index()
        adjacency_matrix = [[0] * max_index for _ in range(max_index)]
        for edge in self.edges:
            adjacency_matrix[edge.node_from.value][edge.node_to.value] \
                = edge.value
        return adjacency_matrix

    def find_max_index(self):
        """Return the highest found node number
        Or the length of the node names if set with set_node_names()."""
        if len(self.node_names):
            return len(self.node_names)
        max_index = -1
        if len(self.nodes):
            for node in self.nodes:
                if node.value > max_index:
                    max_index = node.value
        return max_index + 1

    def get_node(self, node_number) -> Node:
        """Return the node with value node_number or None"""
        return self._node_map.get(node_number)

    def _clear_visited(self):
        for node in self.nodes:
            node.visited = False

    def dfs_helper(self, start_node: Node) -> List[int]:
        """TOD0: Write the helper function for a recursive
        implementation of Depth First Search iterating through
        a node's edges. The output should be a list of numbers
        corresponding to the values of the traversed nodes.
        ARGUMENTS: start_node is the starting Node
        MODIFIES: the value of the visited property of nodes
        in self.nodes
        RETURN: a list of the traversed node values (integers).
        """
        ret_list = [start_node.value]
        # Your code here
        start_node.visited = True
        # edges_out = [edge for edge in start_node.edges
        #              if edge.node_to.value != start_node.value]
        for edge in start_node.edges:
            if not edge.node_to.visited:
                ret_list.extend(self.dfs_helper(edge.node_to))
        return ret_list

    def dfs(self, start_node_num: int) -> List[int]:
        """Outputs a list of numbers corresponding to the traversed
        nodes in a Depth First Search.
        :argument start_node_num is the starting node number
        (integer)
        modifies: the value of the visited property of nodes
        in self.nodes
        :return a list of the node values (integers)"""
        self._clear_visited()
        start_node = self.get_node(start_node_num)
        return self.dfs_helper(start_node)

    def dfs_names(self, start_node_num) -> List[str]:
        """Return the results of dfs with numbers
        converted to names."""
        return [self.node_names[num] for num in self.dfs(start_node_num)]

    def bfs(self, start_node_num) -> List[int]:
        """TOD0: Create an iterative implementation of Breadth First
        Search iterating through a node's edges.The output should be
        a list of numbers corresponding to the traversed nodes.
        :argument start_node_num is the node number (integer)
        modifies: the value of the visited property of nodes in
        self.nodes
        :return a list of the node values (integers)."""
        self._clear_visited()
        ret_list = []
        node = self.get_node(start_node_num)

        queue = [node]
        node.visited = True

        while queue:
            node = queue.pop(0)
            ret_list.append(node.value)
            for edge in node.edges:
                if not edge.node_to.visited:
                    edge.node_to.visited = True
                    queue.append(edge.node_to)

        return ret_list

    def bfs_names(self, start_node_num) -> List[str]:
        """Return the results of bfs with numbers converted to names."""
        return [self.node_names[num] for num in self.bfs(start_node_num)]


if __name__ == '__main__':
    graph = Graph()
    # You do not need to change anything below this line.
    # You only need to implement Graph.dfs_helper and Graph.bfs

    graph.set_node_names(('Mountain View',  # 0
                          'San Francisco',  # 1
                          'London',  # 2
                          'Shanghai',  # 3
                          'Berlin',  # 4
                          'Sao Paolo',  # 5
                          'Bangalore'))  # 6

    graph.insert_edge(51, 0, 1)  # MV <-> SF
    graph.insert_edge(51, 1, 0)  # SF <-> MV
    graph.insert_edge(9950, 0, 3)  # MV <-> Shanghai
    graph.insert_edge(9950, 3, 0)  # Shanghai <-> MV
    graph.insert_edge(10375, 0, 5)  # MV <-> Sao Paolo
    graph.insert_edge(10375, 0, 5)  # Sao Paolo <-> MV
    graph.insert_edge(9900, 1, 3)  # SF <-> Shanghai
    graph.insert_edge(9900, 3, 1)  # Shanghai <-> SF
    graph.insert_edge(9130, 1, 4)  # SF <-> Berlin
    graph.insert_edge(9130, 4, 1)  # Berlin <-> SF
    graph.insert_edge(9217, 2, 3)  # London <-> Shanghai
    graph.insert_edge(9217, 3, 2)  # Shanghai <-> London
    graph.insert_edge(932, 2, 4)  # London <-> Berlin
    graph.insert_edge(932, 4, 2)  # Berlin <-> London
    graph.insert_edge(9471, 2, 5)  # London <-> Sao Paolo
    graph.insert_edge(9471, 5, 2)  # Sao Paolo <-> London
    # (6) 'Bangalore' is intentionally disconnected (no edges)
    # for this problem and should produce None in the
    # Adjacency List, etc.

    pp = pprint.PrettyPrinter(indent=2)

    print("Edge List")
    pp.pprint(graph.get_edge_list_names())

    print("\nAdjacency List")
    pp.pprint(graph.get_adjacency_list_names())

    print("\n Depth First Search")
    pp.pprint(graph.dfs_names(2))

    # Should print:
    # Depth First Search
    # ['London', 'Shanghai', 'Mountain View', 'San Francisco',
    # 'Berlin', 'Sao Paolo']

    print("\nBreadth First Search")
    pp.pprint(graph.bfs_names(2))
    # test error reporting
    # pp.print(['Sao Paulo', 'Mountain View', 'San Francisco',
    # 'London', 'Shanghai', 'Berlin'])

    # Should print:
    # Breadth First Search
    # ['London', 'Shanghai', 'Berlin', 'Sao Paolo',
    # 'Mountain View', 'San Francisco']
