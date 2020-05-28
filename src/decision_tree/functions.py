from collections import OrderedDict

import pandas as pd

from .structure import *


def get_tree(data: pd.DataFrame, times: pd.DataFrame, work_item: str):
    """
    Converts a work items process into a tree object
    :param data: Dataframe with raw data
    :param times: Aggregated dataframe with phase times and frequency from function features.work_times
    :param work_item: Name of the work_item
    :return: Tree object
    """
    values = times.loc[work_item].to_dict()
    tree = Tree()
    tree, nodes = init_nodes(tree, data=values)
    phases = data.groupby("work_item")["phase"].unique().to_dict()
    path = phases[work_item]
    tree, edges = init_edges(tree, nodes, path=path)
    return tree


def init_nodes(tree, data):
    """
    Initializes all nodes with label, time and frequency
    :param tree: Tree object to add nodes to
    :param data: Raw data dataframe
    :return: Tree object with added nodes, dict with nodes
    """
    nodes = OrderedDict()
    node_labels = ["Analyze", "Design", "Build", "Test", "Package", "Accept", "Deploy", *["End"] * 7]
    root = True
    for label in node_labels:
        node = Node({"label": label, "time": data[label], "freq": data[label + "_freq"]})
        nodes[label] = node
        if root is True:
            tree.add_root(node)
            root = False
        else:
            tree.add_node(node)
    return tree, nodes


def init_edges(tree, nodes, path):
    """
    Initializes the edges to a given tree object
    :param tree: The tree you want to add edges
    :param nodes: Nodes dict from init_nodes
    :param path: A list of tuples with the original path to set weights
    :return: Tree object with added edges
    """
    edges = []
    node_labels = ["Analyze", "Design", "Build", "Test", "Package", "Accept", "Deploy", "End"]
    # We iterate over the node_labels
    for i in range(len(nodes) - 1):
        data = {"weight": 0}
        from_node = nodes[node_labels[i]]
        to_node = nodes[node_labels[i + 1]]
        # If path in process set weight = 1
        if len(list(filter(lambda x: node_labels[i] in x[0] and node_labels[i + 1] in x[1], path))) > 0:
            data = {"weight": 1}
        edge1 = Edge(data=data, from_node=from_node, to_node=to_node)
        from_node.add_left(to_node)
        to_node.add_parent(from_node)
        edges.append(edge1)
        tree.add_edge(edge1)
        data = {"weight": 0}
        # Set an edge to the end from every node if current edge not goes to end
        if node_labels[len(node_labels) - 1] not in node_labels[i + 1]:
            end_node = nodes[node_labels[len(node_labels) - 1]]
            if len(list(
                    filter(lambda x: node_labels[i] in x[0] and node_labels[len(node_labels) - 1] in x[1], path))) > 0:
                data = {"weight": 1}
            edge2 = Edge(data=data, from_node=from_node, to_node=end_node)
            from_node.add_right(end_node)
            end_node.add_parent(from_node)
            edges.append(edge2)
            tree.add_edge(edge2)
    return tree, edges
