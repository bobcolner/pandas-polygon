"""
Tests for DualDashGraph class and its functionalities
"""

import os
import unittest

import dash
import dash_cytoscape as cyto
import dash_html_components as html
import pandas as pd
from jupyter_dash import JupyterDash

from mlfinlab.codependence import get_distance_matrix
from mlfinlab.networks.dual_dash_graph import DualDashGraph
from mlfinlab.networks.mst import MST
from mlfinlab.networks.almst import ALMST


class TestDualDashGraph(unittest.TestCase):
    # pylint: disable=protected-access, too-many-public-methods
    """
    Tests for the different DualDashGraph object functionalities and the ALMST functionality.
    """

    def setUp(self):
        """
        Import sample data and create DualDashGraph object for testing.
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

        # Calculate correlation and distances.
        correlation_matrix = log_return_dataframe.corr(method='pearson')
        distance_matrix = get_distance_matrix(correlation_matrix)

        other_correlation_matrix = other_log_return.corr(method='pearson')
        other_distance_matrix = get_distance_matrix(other_correlation_matrix)

        distance_matrix2 = distance_matrix.copy()

        # Creates Graph class objects from matrices.
        self.mst_graph = MST(distance_matrix, "distance")
        # Creates Graph class copy to test for alternative settings.
        mst_graph2 = MST(distance_matrix2, "distance")
        # Graph 3 for JupyterDash testing.
        self.mst_graph3 = MST(distance_matrix, "distance")

        # Creates Graph class objects from matrices.
        self.almst_graph = ALMST(distance_matrix, "distance")
        # Creates Graph class copy to test for alternative settings
        almst_graph2 = ALMST(distance_matrix2, "distance")
        # Graph 3 for JupyterDash testing
        almst_graph3 = ALMST(distance_matrix, "distance")
        # Graph 4 with adifferent dataset
        self.almst_graph4 = ALMST(other_distance_matrix, "distance")

        # Adding industry groups colour assignment
        self.industry = {"stocks": ['EEM', 'EWG', 'EWJ', 'EFA', 'EWQ', 'EWU', 'XLB',
                                    'XLE', 'XLF', 'XLK', 'XLU', 'EPP', 'FXI', 'VGK',
                                    'VPL', 'SPY', 'CSJ'],
                         "bonds": ['TIP', 'IEF', 'LQD', 'TLT', 'BND', 'CSJ', 'DIA']}
        self.mst_graph.set_node_groups(self.industry)
        self.almst_graph.set_node_groups(self.industry)

        # Adding market cap
        self.market_cap = [2000, 2500, 3000, 1000, 5000, 3500, 2000, 2500, 3000, 1000,
                           5000, 3500, 2000, 2500, 3000, 1000, 5000, 3500, 2000, 2500,
                           3000, 1000, 5000]
        self.mst_graph.set_node_size(self.market_cap)
        self.almst_graph.set_node_size(self.market_cap)

        # Create DashGraph object
        self.dash_graph = DualDashGraph(self.mst_graph, self.almst_graph)
        # Create DashGraph object for alternative settings
        self.dash_graph2 = DualDashGraph(mst_graph2, almst_graph2)

        # Test for additional colours on graph 3
        for i in range(17):
            self.industry['category{}'.format(i)] = [i]
        self.mst_graph3.set_node_groups(self.industry)
        almst_graph3.set_node_groups(self.industry)

        # DashGraph for Jupyter Notebook
        self.dash_graph3 = DualDashGraph(self.mst_graph3, almst_graph3, "jupyter notebook")

    def test_select_other_graph_node(self):
        """
        Tests _select_other_graph_node
        """
        # Creating elements to test function
        elements_in = [{"data": {"id": "element1"}},
                       {"data": {"id": "element2"}}]

        data = {"data": {"id": "element2"}}

        # Passing valid parameters
        elements_out = self.dash_graph._select_other_graph_node(data, elements_in)

        # Passing empty data variable
        elements_same = self.dash_graph._select_other_graph_node(None, elements_in)

        # Testing valid output
        self.assertTrue(not elements_out[0]['selected'])
        self.assertTrue(elements_out[1]['selected'])

        # Testing same output
        self.assertTrue(elements_same == elements_in)

    def test_generate_comparison_layout(self):
        """
        Tests _generate_comparison_layout returns html.Div
        """
        comparison_layout = self.dash_graph._generate_comparison_layout(self.mst_graph,
                                                                        self.almst_graph)
        self.assertIsInstance(comparison_layout, html.Div)

    def test_get_default_stylesheet(self):
        """
        Tests correct stylesheet dictionary is returned
        """
        stylesheet = self.dash_graph._get_default_stylesheet(self.dash_graph.one_components[0])
        self.assertEqual(len(stylesheet), 7)

    def test_set_cyto_graph(self):
        """
        Tests generate_layout returns Dash Bootstrap Container
        """
        self.dash_graph._set_cyto_graph()
        self.assertIsInstance(self.dash_graph.cyto_one, cyto.Cytoscape)
        self.assertIsInstance(self.dash_graph.cyto_two, cyto.Cytoscape)

    def test_get_server(self):
        """
        Tests get_server returns a Dash app object
        """
        app = self.dash_graph.get_server()
        self.assertIsInstance(app, dash.Dash)

    def test_jupyter_settings(self):
        """
        Tests JupyterDash app is created instead of Dash app
        """
        expected_elements = ['cytoscape', 'cytoscape_two']

        app3 = self.dash_graph3.get_server()
        self.assertIsInstance(app3, JupyterDash)
        self.assertEqual(app3.serve_layout().status_code, 200)

        for i, item in enumerate(app3.layout):
            self.assertEqual(expected_elements[i], item)

        app2 = self.dash_graph2.get_server()
        self.assertIsInstance(app2, dash.Dash)
        self.assertEqual(app2.serve_layout().status_code, 200)
        for i, item in enumerate(app2.layout):
            self.assertEqual(expected_elements[i], item)

    def test_calculate_average_distance(self):
        """
        Test for the calculate_average_distance method.
        """
        # Create a 3 by 3 matrix for testing.
        dist_array = [[0, 0.2, 0.3], [0.2, 0, 0.4], [0.3, 0.4, 0]]

        distance_df = pd.DataFrame(dist_array)

        # Create clusters
        clusters = [[0, 1], [2]]
        # Set cluster index corresponding to the clusters.
        c_x = 0
        c_y = 1
        average, node_1, node_2 = ALMST._calculate_average_distance(distance_df, clusters, c_x, c_y)

        self.assertEqual(node_2, 2)
        self.assertEqual(node_1, 0)
        average_distance = (0.3 + 0.4)/2
        self.assertEqual(average, average_distance)

    def test_differnet_graphs_error(self):
        """
        Tests that when given two graphs with different set of nodes it results in an error.
        """
        self.assertRaises(ValueError, DualDashGraph, graph_one=self.mst_graph3, graph_two=self.almst_graph4)
