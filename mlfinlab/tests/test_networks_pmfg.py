"""
Tests for PMFG functionality in networks module
"""

import os
import unittest
import json

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot

from mlfinlab.networks.dash_graph import PMFGDash
from mlfinlab.networks.pmfg import PMFG


class TestPMFG(unittest.TestCase):
    # pylint: disable=protected-access
    """
    Tests for PMFG object, and the PMFGDash and its functions in Networks module
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

        # Calculate correlation and distances
        correlation_matrix = log_return_dataframe.corr(method='pearson')
        self.distance_matrix = np.sqrt(2 * (1 - correlation_matrix))

        # Create MST Graph class objects from correlation and distance matrices
        self.pmfg_correlation = PMFG(correlation_matrix, "correlation")
        self.pmfg = PMFG(self.distance_matrix, "distance")

        self.industry = {"stocks": ['EEM', 'EWG', 'EWJ', 'EFA', 'EWQ', 'EWU', 'XLB',
                                    'XLE', 'XLF', 'XLK', 'XLU', 'EPP', 'FXI', 'VGK',
                                    'VPL', 'SPY', 'CSJ'],
                         "bonds": ['TIP', 'IEF', 'LQD', 'TLT', 'BND', 'CSJ', 'DIA']}
        self.market_cap = [2000, 2500, 3000, 1000, 5000, 3500, 2000, 2500, 3000, 1000,
                           5000, 3500, 2000, 2500, 3000, 1000, 5000, 3500, 2000, 2500,
                           3000, 1000, 5000]

        # DashGraph for Jupyter Notebook
        self.dash_jupyter = PMFGDash(self.pmfg, "jupyter notebook")
        self.dash = PMFGDash(self.pmfg)

    def test_mst_styling(self):
        """
        Tests for whether the MST edges are styled correctly.
        """
        edge_list = self.dash.elements[23:]
        mst_edge_count = 0
        for edge_data in edge_list:
            if len(edge_data) > 1 and edge_data['classes'] == 'mst':
                mst_edge_count += 1
        self.assertEqual(mst_edge_count, 22)

    def test_get_mst_edges(self):
        """
        Tests get_mst_edges method for the list of MST edges.
        """
        mst_edges = self.pmfg.get_mst_edges()
        mst_edge_list = {frozenset(x) for x in mst_edges}

        mst_result = [('EEM', 'FXI'), ('EEM', 'EFA'), ('EWG', 'EWQ'), ('TIP', 'IEF'),
                      ('EWJ', 'VPL'), ('EFA', 'VGK'), ('EFA', 'EWU'), ('EFA', 'VPL'),
                      ('EFA', 'SPY'), ('EFA', 'EPP'), ('IEF', 'TLT'), ('IEF', 'BND'),
                      ('EWQ', 'VGK'), ('XLB', 'SPY'), ('XLE', 'SPY'), ('XLF', 'SPY'),
                      ('LQD', 'BND'), ('LQD', 'CSJ'), ('LQD', 'XLU'), ('XLK', 'SPY'),
                      ('XLU', 'SPY'), ('SPY', 'DIA')]
        result_edges = {frozenset(x) for x in mst_result}
        self.assertEqual(set(mst_edge_list), set(result_edges))

    def test_matrix_to_pmfg(self):
        """
        Tests initialisation of NetworkX graphs when given
        distance or correlation matrices
        """
        pmfg_graph = self.pmfg.get_graph()
        # Checking mst has the correct number of edges and nodes
        # 3 (n-2) edges and n nodes for a PMFG
        self.assertEqual(pmfg_graph.number_of_edges(), 63)
        # n - 1 edges in an MST
        self.assertEqual(pmfg_graph.number_of_nodes(), 23)

    def test_matrix_type(self):
        """
        Tests name of matrix type returns as set
        """
        self.assertEqual(self.pmfg.get_matrix_type(), "distance")

    def test_get_pos(self):
        """
        Tests get_pos returns a dictionary of node positions
        """
        pos = self.pmfg.get_pos()
        self.assertEqual(len(pos), 23)
        self.assertIsInstance(pos, dict)
        nodes = ['EEM', 'EWG', 'TIP', 'EWJ', 'EFA', 'IEF', 'EWQ', 'EWU', 'XLB', 'XLE',
                 'XLF', 'LQD', 'XLK', 'XLU', 'EPP', 'FXI', 'VGK', 'VPL', 'SPY', 'TLT',
                 'BND', 'CSJ', 'DIA']
        for i, item in enumerate(pos):
            self.assertEqual(item, nodes[i])

    def test_get_graph(self):
        """
        Tests whether get_graph returns a nx.Graph object and checks if graph is planar.
        """
        self.assertIsInstance(self.pmfg.get_graph(), nx.Graph)
        self.assertEqual(nx.check_planarity(self.pmfg.get_graph())[0], True)

    def test_get_graph_plot(self):
        """
        Tests get_graph returns axes
        """
        axes = self.pmfg.get_graph_plot()
        self.assertIsInstance(axes, pyplot.Axes)

    def test_edge_in_mst(self):
        """
        Tests for whether the edge is in the MST or not.
        """
        mst_edges = [('EEM', 'FXI'), ('EEM', 'EFA'), ('EWG', 'EWQ'), ('TIP', 'IEF'),
                     ('EWJ', 'VPL'), ('EFA', 'VGK'), ('EFA', 'EWU'), ('EFA', 'VPL'),
                     ('EFA', 'SPY'), ('EFA', 'EPP'), ('IEF', 'TLT'), ('IEF', 'BND'),
                     ('EWQ', 'VGK'), ('XLB', 'SPY'), ('XLE', 'SPY'), ('XLF', 'SPY'),
                     ('LQD', 'BND'), ('LQD', 'CSJ'), ('LQD', 'XLU'), ('XLK', 'SPY'),
                     ('XLU', 'SPY'), ('SPY', 'DIA')]
        for edge in mst_edges:
            edge_in_mst = self.pmfg.edge_in_mst(edge[0], edge[1])
            self.assertEqual(edge_in_mst, True)
        non_mst = [('EWU', 'VGK'), ('XLK', 'IEF'), ('XLU', 'XLK'), ('IEF', 'DIA')]
        for edge in non_mst:
            edge_not_mst = self.pmfg.edge_in_mst(edge[0], edge[1])
            self.assertEqual(edge_not_mst, False)

    def test_set_node_group(self):
        """
        Tests industry groups is set as attribute of class
        """
        self.pmfg.set_node_groups(self.industry)
        self.assertEqual(self.pmfg.node_groups, self.industry)
        self.get_node_colours()

    def test_set_node_size(self):
        """
        Tests node size (e.g. market cap) is set as attribute of class
        """
        self.pmfg.set_node_size(self.market_cap)
        self.assertEqual(self.pmfg.node_sizes, self.market_cap)
        self.get_node_sizes()

    def get_node_sizes(self):
        """
        Test for getter method of node sizes
        """
        node_sizes = self.pmfg.get_node_sizes()
        self.assertEqual(node_sizes, self.market_cap)

    def get_node_colours(self):
        """
        Test for getter method of node colours
        """
        node_colours = self.pmfg.get_node_colours()
        self.assertEqual(node_colours, self.industry)

    def test_get_disparity_measure(self):
        """
        Test for getter method of the dictionary of disparity measure values of cliques
        """
        disparity_measure = self.pmfg.get_disparity_measure()
        str_disparity_list = list(disparity_measure.values())
        float_disparity_list = [float(i) for i in str_disparity_list]
        average_disparity = np.mean(float_disparity_list)

        self.assertAlmostEqual(average_disparity, 0.1514982, delta=1e-7)

    def test_update_stat_json(self):
        """
        Checks if Statistics panel returned expected outputs
        """
        # Test for graph summary
        summary = self.dash._update_stat_json("graph_summary")
        self.check_summary_values(json.loads(summary))

        # Test for average_degree_connectivity
        average_degree_connectivity = self.dash._update_stat_json("average_degree_connectivity")
        expected_average_degree = {'7': 7.190476190476191, '3': 7.833333333333333,
                                   '5': 6.666666666666667, '12': 5.666666666666667,
                                   '4': 7.05, '6': 6.611111111111111,
                                   '9': 6.888888888888889, '13': 5.846153846153846}

        self.assertEqual(json.loads(average_degree_connectivity), expected_average_degree)

        # Test for average_neighbor_degree
        average_neighbor_degree = self.dash._update_stat_json("average_neighbor_degree")
        expected_average_neighbor = {'EEM': 7.0, 'EWG': 8.333333333333334, 'TIP': 5.0,
                                     'EWJ': 10.333333333333334, 'EFA': 5.666666666666667,
                                     'IEF': 5.25, 'EWQ': 7.25, 'EWU': 8.4, 'XLB': 8.25,
                                     'XLE': 9.666666666666666, 'XLF': 8.0, 'LQD': 6.857142857142857,
                                     'XLK': 7.5, 'XLU': 7.833333333333333, 'EPP': 7.0,
                                     'FXI': 5.666666666666667, 'VGK': 6.888888888888889,
                                     'VPL': 7.0, 'SPY': 5.846153846153846, 'TLT': 5.0,
                                     'BND': 5.0, 'CSJ': 6.6, 'DIA': 7.714285714285714}
        self.assertEqual(json.loads(average_neighbor_degree), expected_average_neighbor)

        # Test for betweenness_centrality
        betweenness_centrality = self.dash._update_stat_json("betweenness_centrality")
        expected_betweenness = {"EEM": 0.06597577506668416, "EWG": 0.0,
                                "TIP": 0.02191175827539464, "EWJ": 0.0,
                                "EFA": 0.17626918536009448, "IEF": 0.015096856005946917,
                                "EWQ": 0.001443001443001443, "EWU": 0.0075036075036075045,
                                "XLB": 0.001443001443001443, "XLE": 0.0,
                                "XLF": 0.0, "LQD": 0.20580480125934672,
                                "XLK": 0.003246753246753247, "XLU": 0.07845992391446938,
                                "EPP": 0.00804145349599895, "FXI": 0.0,
                                "VGK": 0.15099698281516463, "VPL": 0.042885565612838336,
                                "SPY": 0.36598889326162054, "TLT": 0.0,
                                "BND": 0.053539726266998995, "CSJ": 0.02585596221959858,
                                "DIA": 0.04826402553675281}
        self.assertEqual(json.loads(betweenness_centrality), expected_betweenness)

        # Test for disparity_measure
        disparity_measure = self.dash._update_stat_json("disparity_measure")
        disparity_measure = json.loads(disparity_measure)
        str_disparity_list = list(disparity_measure.values())
        float_disparity_list = [float(i) for i in str_disparity_list]
        average_disparity = np.mean(float_disparity_list)

        self.assertAlmostEqual(average_disparity, 0.1514982, delta=1e-7)

    def check_summary_values(self, summary):
        """
        Checks graph summary dictionary
        """
        self.assertEqual(summary['nodes'], 23)
        self.assertEqual(summary['edges'], 63)
        self.assertEqual(summary['smallest_edge'], 0.2149)
        self.assertEqual(summary['largest_edge'], 1.407)
        self.assertEqual(summary['normalised_tree_length'], 0.6427285714285712)
        self.assertEqual(summary['average_node_connectivity'], 3.225296442687747)
        self.assertEqual(summary['average_shortest_path'], 2.1620553359683794)
