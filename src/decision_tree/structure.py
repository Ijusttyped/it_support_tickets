# TODO add a plot function for tree class
from pathlib import Path

from graphviz import Digraph


class Tree:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.root = None
        self.leaf = None
        # self.depth = len(nodes)

    def add_root(self, node):
        self.root = node
        self.nodes.append(node)

    def add_leaf(self, node):
        self.leaf = node
        self.nodes.append(node)

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_depth(self, node):
        return self.nodes.index(node)

    def plot(self, name, path):
        g = Digraph(name=name, directory=Path(path), node_attr={'height': '.1'})
        for node in self.nodes:
            g.node(node.get_label())
        for edge in self.edges:
            g.edge(tail_name=edge.get_fromnode(), head_name=edge.get_tonode(), label=str(edge.get_weight()))
        g.attr(label=name)
        g.render()
        return g


class Node:
    def __init__(self, data):
        self.data = data
        self.parent = None
        self.left_child = None
        self.right_child = None

    def add_parent(self, node):
        self.parent = node

    def add_left(self, node):
        self.left_child = node

    def add_right(self, node):
        self.right_child = node

    def get_label(self):
        return str(self.data).replace(":", "=")


class Edge:
    def __init__(self, data, from_node, to_node):
        self.data = data
        self.from_node = from_node
        self.to_node = to_node

    def add_fromnode(self, node):
        self.from_node = node

    def get_fromnode(self):
        return str(self.from_node.data).replace(":", "=")

    def add_tonode(self, node):
        self.to_node = node

    def get_tonode(self):
        return str(self.to_node.data).replace(":", "=")

    def get_weight(self):
        return self.data["weight"]

    def prints(self):
        return (self.from_node.data, self.to_node.data)
