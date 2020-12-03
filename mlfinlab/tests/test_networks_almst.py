"""
Tests for Graph class in networks module
"""

import os
import unittest

import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot

from mlfinlab.networks.almst import ALMST


class TestGraph(unittest.TestCase):
    # pylint: disable=protected-access
    """
    Tests for Graph object and its functions in Networks module
    """

    def setUp(self):
        """
        Set up path to import test data
        """
        # Set project path to current directory.
        project_path = os.path.dirname(__file__)
        # Add new data path to match stock_prices.csv data.
        data_path = project_path + '/test_data/stock_prices.csv'
        log_return_dataframe = pd.read_csv(data_path, index_col=0)
        log_return_dataframe = log_return_dataframe.pct_change()

        other_data_path = project_path + '/test_data/stock_prices_2.csv'
        other_log_return = pd.read_csv(other_data_path, index_col=0)
        other_log_return = other_log_return.pct_change()

        # Calculate correlation and distances
        correlation_matrix = log_return_dataframe.corr(method='pearson')
        self.distance_matrix = np.sqrt(2 * (1 - correlation_matrix))

        other_correlation_matrix = other_log_return.corr(method='pearson')
        self.other_distance_matrix = np.sqrt(2 * (1 - other_correlation_matrix))

        # Create ALMST Graph class objects from distance matrix using Kruskal's algorithm
        self.almst_graph_kruskal = ALMST(self.distance_matrix, matrix_type="distance",
                                         mst_algorithm="kruskal")

        # Create ALMST Graph class objects from correlation matrix using Prim's algorithm
        self.almst_graph_prim = ALMST(other_correlation_matrix, matrix_type="correlation",
                                      mst_algorithm="prim")

        self.industry = {"stocks": ['EEM', 'EWG', 'EWJ', 'EFA', 'EWQ', 'EWU', 'XLB',
                                    'XLE', 'XLF', 'XLK', 'XLU', 'EPP', 'FXI', 'VGK',
                                    'VPL', 'SPY', 'CSJ'],
                         "bonds": ['TIP', 'IEF', 'LQD', 'TLT', 'BND', 'CSJ', 'DIA']}
        self.market_cap = [2000, 2500, 3000, 1000, 5000, 3500, 2000, 2500, 3000, 1000,
                           5000, 3500, 2000, 2500, 3000, 1000, 5000, 3500, 2000, 2500,
                           3000, 1000, 5000]

    def test_invalid_almst_algorithm(self):
        """
        Tests for invalid ALMST algorithm type which raises a ValueError.
        """
        self.assertRaises(ValueError, ALMST, self.distance_matrix, "distance", mst_algorithm="invalid algo")

    def test_matrix_to_mst(self):
        """
        Tests initialisation of NetworkX graphs when given
        distance or correlation matrices
        """
        almst_distance_graph = self.almst_graph_kruskal.get_graph()
        almst_correlation_graph = self.almst_graph_prim.get_graph()

        # Checking mst has the correct number of edges and nodes
        self.assertEqual(almst_distance_graph.number_of_edges(), 22)
        self.assertEqual(almst_distance_graph.number_of_nodes(), 23)

        self.assertEqual(almst_correlation_graph.number_of_edges(), 5)
        self.assertEqual(almst_correlation_graph.number_of_nodes(), 6)

    def test_matrix_type(self):
        """
        Tests name of matrix type returns as set
        """
        self.assertEqual(self.almst_graph_kruskal.get_matrix_type(), "distance")
        self.assertEqual(self.almst_graph_prim.get_matrix_type(), "correlation")

    def test_get_pos(self):
        """
        Tests get_pos returns a dictionary of node positions
        """
        pos_distance = self.almst_graph_kruskal.get_pos()
        pos_correlation = self.almst_graph_prim.get_pos()

        self.assertEqual(len(pos_distance), 23)
        self.assertIsInstance(pos_distance, dict)

        self.assertEqual(len(pos_correlation), 6)
        self.assertIsInstance(pos_correlation, dict)

        nodes = ['EEM', 'EWG', 'TIP', 'EWJ', 'EFA', 'IEF', 'EWQ', 'EWU', 'XLB', 'XLE',
                 'XLF', 'LQD', 'XLK', 'XLU', 'EPP', 'FXI', 'VGK', 'VPL', 'SPY', 'TLT',
                 'BND', 'CSJ', 'DIA']

        nodes_small = ['Apple', 'Amazon', 'Facebook', 'Microsoft', 'Netflix', 'Tesla']

        for i, item in enumerate(pos_distance):
            self.assertEqual(item, nodes[i])

        for i, item in enumerate(pos_correlation):
            self.assertEqual(item, nodes_small[i])

    def test_get_graph(self):
        """
        Tests whether get_graph returns a nx.Graph object
        """
        self.assertIsInstance(self.almst_graph_kruskal.get_graph(), nx.Graph)
        self.assertIsInstance(self.almst_graph_prim.get_graph(), nx.Graph)

    def test_get_graph_plot(self):
        """
        Tests get_graph returns axes
        """
        axes_distance = self.almst_graph_kruskal.get_graph_plot()
        axes_correlation = self.almst_graph_prim.get_graph_plot()

        self.assertIsInstance(axes_distance, pyplot.Axes)
        self.assertIsInstance(axes_correlation, pyplot.Axes)

    def test_calculate_average_distance(self):
        """
        Test for the calculate_average_distance method.
        """
        # Create a 3 by 3 matrix for testing.
        dist_array = [[0, 0.2, 0.3], [0.2, 0, 0.4], [0.3, 0.4, 0]]
        matrix = pd.DataFrame(dist_array)
        # Create clusters
        clusters = [[0, 1], [2]]
        # Set cluster index corresponding to the clusters.
        c_x = 0
        c_y = 1
        average, node_1, node_2 = ALMST._calculate_average_distance(matrix, clusters, c_x, c_y)
        self.assertEqual(node_2, 2)
        self.assertEqual(node_1, 0)
        average_distance = (0.3 + 0.4) / 2
        self.assertEqual(average, average_distance)

    def test_set_node_group(self):
        """
        Tests industry groups is set as attribute of class
        """
        self.almst_graph_kruskal.set_node_groups(self.industry)
        self.almst_graph_prim.set_node_groups(self.industry)

        self.assertEqual(self.almst_graph_kruskal.node_groups, self.industry)
        self.assertEqual(self.almst_graph_prim.node_groups, self.industry)

        self.get_node_colours()

    def test_set_node_size(self):
        """
        Tests node size (e.g. market cap) is set as attribute of class
        """
        self.almst_graph_kruskal.set_node_size(self.market_cap)
        self.almst_graph_prim.set_node_size(self.market_cap)

        self.assertEqual(self.almst_graph_kruskal.node_sizes, self.market_cap)
        self.assertEqual(self.almst_graph_prim.node_sizes, self.market_cap)

        self.get_node_sizes()

    def get_node_sizes(self):
        """
        Test for getter method of node sizes
        """
        sizes_distance = self.almst_graph_kruskal.get_node_sizes()
        sizes_correlation = self.almst_graph_prim.get_node_sizes()

        self.assertEqual(sizes_distance, self.market_cap)
        self.assertEqual(sizes_correlation, self.market_cap)

    def get_node_colours(self):
        """
        Test for getter method of node colours
        """
        colours_distance = self.almst_graph_kruskal.get_node_colours()
        colours_correlation = self.almst_graph_prim.get_node_colours()

        self.assertEqual(colours_distance, self.industry)
        self.assertEqual(colours_correlation, self.industry)
