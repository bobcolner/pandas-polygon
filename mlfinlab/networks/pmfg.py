# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This file contains Graph classes, which create NetworkX's Graph objects from matrices.
"""

import heapq
import itertools
from itertools import count
import warnings

import networkx as nx
from matplotlib import pyplot as plt

from mlfinlab.networks.graph import Graph


class PMFG(Graph):
    """
    PMFG class creates and stores the PMFG as an attribute.
    """

    def __init__(self, input_matrix, matrix_type):
        """
        PMFG class creates the Planar Maximally Filtered Graph and stores it as an attribute.

        :param input_matrix: (pd.Dataframe) Input distance matrix
        :param matrix_type: (str) Matrix type name (e.g. "distance").
        """
        super().__init__(matrix_type)

        # To store the MST edges
        self.mst_edges = []

        # Create a nx.Graph object from ALSMT dataframe
        self.graph = self.create_pmfg(input_matrix)

        # Create positions
        with warnings.catch_warnings():  # Silencing specific FutureWarning
            warnings.filterwarnings('ignore', r'arrays to stack must be passed as a "sequence"')
            self.pos = nx.planar_layout(self.graph)

        # Attribute to store 3 cliques and 4 cliques
        self.three_cliques = []
        self.four_cliques = []

        # Generate the cliques
        self._generate_cliques()

        # Calculate disparity measure for the cliques
        self.disparity = self._calculate_disparity()

    def get_disparity_measure(self):
        """
        Getter method for the dictionary of disparity measure values of cliques.

        :return: (Dict) Returns a dictionary of clique to the disparity measure.
        """
        return self.disparity

    def _calculate_disparity(self):
        """
        Calculate disparity given in Tumminello M, Aste T, Di Matteo T, Mantegna RN.
        A tool for filtering information in complex systems.
        https://arxiv.org/pdf/cond-mat/0501335.pdf

        :return: (Dict) Returns a dictionary of clique to the disparity measure.
        """
        disparity_dict = {}
        for clique in self.three_cliques + self.four_cliques:
            for i in clique:
                strength_i = 0
                summ = 0
                for j in clique:
                    if i != j:
                        strength_i += self.graph[i][j]['weight']
                for j in clique:
                    if i != j:
                        disparity = (self.graph[i][j]['weight'] /
                                     strength_i)**2 if strength_i else 0
                        summ += disparity
            disparity_dict[str(clique)] = summ / len(clique)
        return disparity_dict

    def _generate_cliques(self):
        """
        Generate cliques from all of the nodes in the PMFG.
        """
        all_cliques = nx.algorithms.enumerate_all_cliques(self.graph)
        for clique in all_cliques:
            if len(clique) == 3:
                self.three_cliques.append(clique)
            if len(clique) == 4:
                self.four_cliques.append(clique)

    def create_pmfg(self, input_matrix):
        """
        Creates the PMFG matrix from the input matrix of all edges.

        :param input_matrix: (pd.Dataframe) Input matrix with all edges
        :return: (nx.Graph) Output PMFG matrix
        """
        nodes = list(input_matrix.columns)

        # Heap to store ordered edges
        heap = []
        cnt = count()

        # Generate pairwise combination between nodes
        pairwise_combinations = list(itertools.combinations(range(len(nodes)), 2))

        # For cluster 1 and cluster 2 in the pairwise combination lists
        for i, j in pairwise_combinations:
            node_i = nodes[i]
            node_j = nodes[j]
            weight = input_matrix[node_i][node_j]

            # If the matrix type is correlation, order the edge list by largest to smallest
            if self.matrix_type == "correlation":
                heapq.heappush(heap, (weight*-1.0, next(cnt), {'node_i': node_i, 'node_j': node_j}))
            else:
                heapq.heappush(heap, (weight, next(cnt), {'node_i': node_i, 'node_j': node_j}))

        # Add the nodes with no edges to the PMFG graph
        pmfg_graph = nx.Graph()
        for node in nodes:
            pmfg_graph.add_node(node)

        # Starting with the smallest, keep adding edges
        while len(pmfg_graph.edges()) != 3 * (len(nodes) -2):
            _, _, edge = heapq.heappop(heap)
            node_i = edge['node_i']
            node_j = edge['node_j']
            weight = input_matrix.at[node_i, node_j]
            pmfg_graph.add_weighted_edges_from([(node_i, node_j, weight)])
            if not nx.check_planarity(pmfg_graph)[0]:
                pmfg_graph.remove_edge(node_i, node_j)

        # Store the MST edges as an attribute, so we can style those edges differently.
        self.mst_edges = nx.minimum_spanning_tree(pmfg_graph).edges()

        return pmfg_graph

    def get_mst_edges(self):
        """
        Returns the list of MST edges.

        :return: (list) Returns a list of tuples of edges.
        """
        return self.mst_edges

    def edge_in_mst(self, node1, node2):
        """
        Checks whether the edge from node1 to node2 is a part of the MST.

        :param node1: (str) Name of the first node in the edge.
        :param node2: (str) Name of the second node in the edge.
        :return: (bool) Returns true if the edge is in the MST. False otherwise.
        """
        s_edges = {frozenset((node_i, node_j)) for node_i, node_j in self.mst_edges}
        edge = frozenset((node1, node2))
        in_mst = edge in s_edges
        return in_mst

    def get_graph_plot(self):
        """
        Overrides parent get_graph_plot to plot it in a planar format.

        Returns the graph of the MST with labels.
        Assumes that the matrix contains stock names as headers.

        :return: (AxesSubplot) Axes with graph plot. Call plt.show() to display this graph.
        """
        cmap = plt.cm.Blues
        num_edges = len(self.graph.edges(data=True))
        _, axes = plt.subplots(figsize=(12, 6))
        axes.set_title("Minimum Spanning Tree")

        with warnings.catch_warnings():  # Silencing specific FutureWarning
            warnings.filterwarnings('ignore', r'arrays to stack must be passed as a "sequence"')
            nx.draw_planar(self.graph, with_labels=True, edge_color=range(num_edges), edge_cmap=cmap)
        return axes
