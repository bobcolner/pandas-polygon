"""
Tests for DashGraph class and its functionalities
"""

import json
import os
import unittest

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_html_components as html
import pandas as pd
from jupyter_dash import JupyterDash

from mlfinlab.networks.dash_graph import DashGraph
from mlfinlab.networks.mst import MST


class TestDashGraph(unittest.TestCase):
    # pylint: disable=protected-access, too-many-public-methods
    """
    Tests for the different DashGraph object functionalities
    """

    def setUp(self):
        """
        Import sample data and create DashGraph object for testing
        """
        # Set project path to current directory.
        project_path = os.path.dirname(__file__)
        # Add new data path to match stock_prices.csv data.
        data_path = project_path + '/test_data/stock_prices_2.csv'
        log_return_dataframe = pd.read_csv(data_path, index_col=False)

        # Calculate correlation and distances
        correlation_matrix = log_return_dataframe.corr(method='pearson')

        # Correlation matrix copy for alternative settings
        correlation_matrix2 = correlation_matrix.copy()

        # Creates Graph class objects from matrices
        self.graph = MST(correlation_matrix, "correlation")
        # Creates Graph class copy to test for alternative settings
        self.graph2 = MST(correlation_matrix2, "correlation")
        # Graph 3 for JupyterDash testing
        self.graph3 = MST(correlation_matrix, "correlation")

        # Adding industry groups colour assignment
        industry = {"tech": ['Apple', 'Amazon', 'Facebook'],
                    "utilities": ['Microsoft', 'Netflix'],
                    "automobiles": ['Tesla']}
        self.graph.set_node_groups(industry)

        # Adding market cap
        market_caps = [2000, 2500, 3000, 1000, 5000, 3500]
        self.graph.set_node_size(market_caps)

        # Create DashGraph object
        self.dash_graph = DashGraph(self.graph)
        # Create DashGraph object for alternative settings
        self.dash_graph2 = DashGraph(self.graph2)

        # Test for additional colours on graph 3
        for i in range(17):
            industry['category{}'.format(i)] = [i]
        self.graph3.set_node_groups(industry)

        # DashGraph for Jupyter Notebook
        self.dash_graph3 = DashGraph(self.graph3, "jupyter notebook")

    def test_additional_colours(self):
        """
        If more than 19 groups are added, test random colours are assigned
        """
        colours_groups = self.dash_graph3.colour_groups
        self.assertEqual(len(colours_groups), 20)

    def test_jupyter_settings(self):
        """
        Tests JupyterDash app is created instead of Dash app
        """
        expected_elements = ['dropdown-layout', 'dropdown-stat', 'json-output',
                             'rounding_decimals', 'card-content', 'cytoscape', 'positioned-toast']

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

    def test_colours_unassigned(self):
        """
        Tests colour groups are unassigned when no colours are set
        """
        colour_groups_attribute = self.dash_graph2.colour_groups
        self.assertEqual(colour_groups_attribute, {})

    def test_size_unassigned(self):
        """
        Tests sizes are unassigned when no sizes are set
        """
        stylesheet_length = len(self.dash_graph2.stylesheet)
        default_stylesheet_length = len(self.dash_graph2._get_default_stylesheet())
        self.assertEqual(stylesheet_length, default_stylesheet_length)

    def test_assign_colours_to_groups(self):
        """
        Tests correct assignment of colours to industry groups
        """
        colour_groups_map = self.dash_graph.colour_groups
        expected_colour_groups = {
            "tech": "#d0b7d5",
            "utilities": "#a0b3dc",
            "automobiles": "#90e190"
        }
        self.assertEqual(colour_groups_map, expected_colour_groups)

    def test_style_colours(self):
        """
        Checks style sheet has been appended with colour styles
        """
        current_stylesheet = len(self.dash_graph.stylesheet)
        self.dash_graph._style_colours()
        appended_stylesheet = len(self.dash_graph.stylesheet)
        self.assertEqual(appended_stylesheet, current_stylesheet+3)

    def test_get_node_group(self):
        """
        Checks correct industry group returned when index inputted
        """
        node1 = self.dash_graph._get_node_group('Apple')
        self.assertEqual(node1, "tech")
        node2 = self.dash_graph._get_node_group('Amazon')
        self.assertEqual(node2, "tech")
        node3 = self.dash_graph._get_node_group('Facebook')
        self.assertEqual(node3, "tech")
        node4 = self.dash_graph._get_node_group('Microsoft')
        self.assertEqual(node4, "utilities")
        node6 = self.dash_graph._get_node_group('Tesla')
        self.assertEqual(node6, "automobiles")
        # Test for when object not assigned to group
        node_nan = self.dash_graph._get_node_group('invalid name')
        self.assertEqual(node_nan, "default")

    def test_get_node_size(self):
        """
        Checks correct market cap node size returned when index inputted
        """
        node1_size = self.dash_graph._get_node_size(0)
        self.assertEqual(node1_size, 2000)
        node2_size = self.dash_graph._get_node_size(1)
        self.assertEqual(node2_size, 2500)
        node3_size = self.dash_graph._get_node_size(2)
        self.assertEqual(node3_size, 3000)
        node8_size = self.dash_graph._get_node_size(5)
        self.assertEqual(node8_size, 3500)
        # No sizes have been assigned
        node_nan = self.dash_graph2._get_node_size(3)
        self.assertEqual(node_nan, 0)

    def test_assign_sizes(self):
        """
        Checks style sheet has been appended with node sizes
        """
        current_stylesheet = len(self.dash_graph.stylesheet)
        self.dash_graph._assign_sizes()
        appended_stylsheet = len(self.dash_graph.stylesheet)
        self.assertEqual(appended_stylsheet, current_stylesheet+1)

    def test_get_server(self):
        """
        Tests get_server returns a Dash app object
        """
        app = self.dash_graph.get_server()
        self.assertIsInstance(app, dash.Dash)

    def test_generate_layout(self):
        """
        Tests generate_layout returns Dash Bootstrap Container
        """
        layout = self.dash_graph._generate_layout()
        self.assertIsInstance(layout, dbc.Container)

    def test_set_cyto_graph(self):
        """
        Tests generate_layout returns Dash Bootstrap Container
        """
        self.dash_graph._set_cyto_graph()
        self.assertIsInstance(self.dash_graph.cyto_graph, cyto.Cytoscape)

    def test_update_cytoscape_layout(self):
        """
        Tests correct layout is returned
        """
        layout = self.dash_graph._update_cytoscape_layout("cola")
        self.assertEqual(layout['name'], "cola")

    def test_update_stat_json(self):
        """
        Checks if Statistics panel returned expected outputs
        """
        # Test for graph summary
        summary = self.dash_graph._update_stat_json("graph_summary")
        self.check_summary_values(json.loads(summary))

        # Test for average_degree_connectivity
        average_degree_connectivity = self.dash_graph._update_stat_json("average_degree_connectivity")
        expected_average_degree = {'1': 5.0, '5': 1.0}
        self.assertEqual(json.loads(average_degree_connectivity), expected_average_degree)

        # Test for average_neighbor_degree
        average_neighbor_degree = self.dash_graph._update_stat_json("average_neighbor_degree")
        expected_average_neighbor = {'Apple': 5.0, 'Amazon': 5.0, 'Facebook': 5.0,
                                     'Microsoft': 5.0, 'Netflix': 5.0, 'Tesla': 1.0}
        self.assertEqual(json.loads(average_neighbor_degree), expected_average_neighbor)

        # Test for betweenness_centrality
        betweenness_centrality = self.dash_graph._update_stat_json("betweenness_centrality")
        expected_betweenness = {'Apple': 0.0, 'Amazon': 0.0, 'Facebook': 0.0,
                                'Microsoft': 0.0, 'Netflix': 0.0, 'Tesla': 1.0}
        self.assertEqual(json.loads(betweenness_centrality), expected_betweenness)

    def test_get_graph_summary(self):
        """
        Tests if graph summary values are returned correctly
        """
        summary = self.dash_graph.get_graph_summary()
        self.check_summary_values(summary)

    def check_summary_values(self, summary):
        """
        Checks graph summary dictionary
        """
        self.assertEqual(summary['nodes'], 6)
        self.assertEqual(summary['edges'], 5)
        self.assertEqual(summary['smallest_edge'], 0.4706)
        self.assertEqual(summary['largest_edge'], 0.6965)
        self.assertEqual(summary['normalised_tree_length'], 0.5525800000000001)
        self.assertEqual(summary['average_node_connectivity'], 1.0)
        self.assertEqual(summary['average_shortest_path'], 1.6666666666666667)

    def test_round_decimals(self):
        """
        Tests decimals are rounded to desired decimal place
        when round_decimals method is called
        """
        elements = self.dash_graph._round_decimals(None)
        self.assertIsInstance(elements, (list,))
        self.dash_graph._round_decimals(1)
        for weight in self.dash_graph.weights:
            decimal_places = len(str(weight).split('.')[1])
            self.assertEqual(decimal_places, 1)

    def test_get_toast(self):
        """
        Tests Div containing Toast (Bootstrap component) is returned
        """
        toast_div = self.dash_graph._get_toast()
        self.assertIsInstance(toast_div, html.Div)
        toast = toast_div.children[0]
        self.assertIsInstance(toast, dbc.Toast)

    def test_get_default_controls(self):
        """
        Tests Dash Bootstrap component is returned
        """
        controls = self.dash_graph._get_default_controls()
        self.assertIsInstance(controls, dbc.Card)

    def test_get_default_stylesheet(self):
        """
        Tests correct stylesheet dictionary is returned
        """
        stylesheet = self.dash_graph._get_default_stylesheet()
        self.assertEqual(len(stylesheet), 4)
