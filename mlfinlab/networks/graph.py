# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This file contains Graph classes, which create NetworkX's Graph objects from matrices.
"""

from abc import ABC

import networkx as nx
from matplotlib import pyplot as plt


class Graph(ABC):
    """
    This Graph class is a parent class for different types of graphs such as a MST.
    """

    def __init__(self, matrix_type):
        """
        Initializes the Graph object and the Graph class attributes.
        This includes the specific graph such as a MST stored as an attribute.

        :param matrix_type: (str) Name of the matrix type (e.g. "distance" or "correlation").
        """
        # The MST is stored in this variable
        self.graph = None
        # Dictionary of node positions
        self.pos = None
        # Matrix input type e.g. distance, correlation
        self.matrix_type = matrix_type
        # Node groups for colouring nodes according to the industry
        self.node_groups = {}
        # Node sizes for sizing nodes according to the market cap
        self.node_sizes = []

    def get_matrix_type(self):
        """
        Returns the matrix type set at initialisation.

        :return: (str) String of matrix type (eg. "correlation" or "distance").
        """
        return self.matrix_type

    def get_graph(self):
        """
        Returns the Graph stored as an attribute.

        :return: (nx.Graph) Returns a NetworkX graph object.
        """
        return self.graph

    def get_difference(self, input_graph_two):
        """
        Given two Graph with the same nodes, return a set of differences in edge connections.

        :param input_graph_two: (Graph) A graph to compare self.graph against.
        :return: (List) A list of unique tuples showing different edge connections.
        """
        graph_one = self.graph
        graph_two = input_graph_two.get_graph()

        # Create empty graph
        empty_graph = nx.create_empty_copy(graph_one)
        if set(graph_one) != set(graph_two):
            msg = "Graph one does not have the same nodes as graph two."
            raise ValueError(msg)

        # Create frozen set to make a unique set of edges
        r_edges = {frozenset(x) for x in graph_one.edges()}
        s_edges = {frozenset(x) for x in graph_two.edges()}

        # List of differences
        diff_edges = r_edges.symmetric_difference(s_edges)

        # Add different edges to empty graph
        empty_graph.add_edges_from(diff_edges)

        return empty_graph.edges()

    def get_pos(self):
        """
        Returns the dictionary of the nodes coordinates.

        :return: (Dict) Dictionary of node coordinates.
        """
        return self.pos

    def get_graph_plot(self):
        """
        Returns the graph of the MST with labels.
        Assumes that the matrix contains stock names as headers.

        :return: (AxesSubplot) Axes with graph plot. Call plt.show() to display this graph.
        """
        cmap = plt.cm.Blues
        num_edges = len(self.graph.edges(data=True))
        _, axes = plt.subplots(figsize=(12, 6))
        axes.set_title("Minimum Spanning Tree")
        nx.draw(self.graph, self.pos, with_labels=True, edge_color=range(num_edges), edge_cmap=cmap)
        return axes

    def set_node_groups(self, industry_groups):
        """
        Sets the node industry group, by taking in a dictionary of industry group to a list of node indexes.

        :param industry_groups: (Dict) Dictionary of the industry name to a list of node indexes.
        """
        self.node_groups = industry_groups

    def set_node_size(self, market_caps):
        """
        Sets the node sizes, given a list of market cap values corresponding to node indexes.

        :param market_caps: (List) List of numbers corresponding to node indexes.
        """
        self.node_sizes = market_caps

    def get_node_sizes(self):
        """
        Returns the node sizes as a list.

        :return: (List) List of numbers representing node sizes.
        """
        return self.node_sizes

    def get_node_colours(self):
        """
        Returns a map of industry group matched with list of nodes.

        :return: (Dict) Dictionary of industry name to list of node indexes.
        """
        return self.node_groups
