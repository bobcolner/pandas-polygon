"""
Tests for visualisations.py functions which help to easily deploy Dash servers.
"""

import os
import unittest

import dash
import numpy as np
import pandas as pd
from jupyter_dash import JupyterDash

from mlfinlab.networks.pmfg import PMFG
from mlfinlab.networks.visualisations import (generate_mst_server, create_input_matrix,
                                              generate_almst_server, generate_mst_almst_comparison,
                                              generate_pmfg_server, generate_central_peripheral_ranking)


class TestVisualisations(unittest.TestCase):
    """
    Tests for the different options in the visualisations.py methods.
    """

    def setUp(self):
        """
        Sets up the data input path.
        """
        # Set project path to current directory.
        project_path = os.path.dirname(__file__)
        # Add new data path to match stock_prices.csv data.
        data_path = project_path + '/test_data/stock_prices_2.csv'
        self.log_return_dataframe = pd.read_csv(data_path, index_col=False)

    def test_create_input_matrix(self):
        """
        Tests distance matrix sub type inputs.
        """
        input_matrix = create_input_matrix(self.log_return_dataframe, 'angular')
        self.check_angular_distance(input_matrix)
        # An incorrect sub type raises Value Error
        self.assertRaises(ValueError, create_input_matrix, self.log_return_dataframe, 'invalid matrix subtype')

    def check_angular_distance(self, input_matrix):
        """
        Tests angular distance correctly returned when 'angular' is passed as a parameter.
        """
        # Check structure of matrix is correct
        stocks = ['Apple', 'Amazon', 'Facebook', 'Microsoft', 'Netflix', 'Tesla']
        # Check stock names are matrix indexes
        self.assertEqual(list(input_matrix.index.values), stocks)
        self.assertEqual(list(input_matrix.columns.values), stocks)
        # Make sure the diagonal matrix
        self.assertEqual(sum(np.diag(input_matrix)), 0)

        # Check values of the matrix are correct
        self.assertEqual(input_matrix.iat[0, 1], 0.2574781544131463)
        self.assertEqual(input_matrix.iat[1, 2], 0.24858600121487132)
        self.assertEqual(input_matrix.iat[0, 4], 0.2966985001647295)
        self.assertEqual(input_matrix.iat[1, 4], 0.12513992168768526)
        self.assertEqual(input_matrix.iat[2, 3], 0.2708412819346416)

    def test_default_generate_mst_server(self):
        """
        Tests the default, minimal input of the method generate_mst_server.
        """
        default_server = generate_mst_server(self.log_return_dataframe)
        self.assertIsInstance(default_server, dash.Dash)
        for element in default_server.layout['cytoscape'].elements:
            if len(element) > 1:
                colour_group = element['data']['colour_group']
                self.assertEqual(colour_group, 'default')
                size = element['data']['size']
                self.assertEqual(size, 0)

    def test_jupyter_generate_mst_server(self):
        """
        Tests the Jupyter notebook option for the generator method.
        """
        jupyter_server = generate_mst_server(self.log_return_dataframe, jupyter=True)
        self.assertIsInstance(jupyter_server, JupyterDash)

    def test_colours_mst_server(self):
        """
        Tests the groups are added correctly, when they are passed using the colours parameter.
        """
        colours_input = {"tech": ['Apple', 'Amazon', 'Facebook', 'Microsoft', 'Netflix', 'Tesla']}
        server_colours = generate_mst_server(self.log_return_dataframe, colours=colours_input)
        for element in server_colours.layout['cytoscape'].elements:
            if len(element) > 1:
                colour_group = element['data']['colour_group']
                self.assertEqual(colour_group, 'tech')

    def test_sizes_mst_server(self):
        """
        Tests the sizes are added correctly when sizes are passed in as a parameter.
        """
        sizes_input = [100, 240, 60, 74, 22, 111]
        server_sizes = generate_mst_server(self.log_return_dataframe, sizes=sizes_input)
        sizes_output = []
        for element in server_sizes.layout['cytoscape'].elements:
            if len(element) > 1:
                size = element['data']['size']
                sizes_output.append(size)
        self.assertEqual(sizes_output, sizes_input)

    def test_default_generate_almst_server(self):
        """
        Tests the default, minimal input of the method generate_almst_server.
        """
        default_server = generate_almst_server(self.log_return_dataframe)
        self.assertIsInstance(default_server, dash.Dash)
        for element in default_server.layout['cytoscape'].elements:
            if len(element) > 1:
                colour_group = element['data']['colour_group']
                self.assertEqual(colour_group, 'default')
                size = element['data']['size']
                self.assertEqual(size, 0)

    def test_jupyter_generate_almst_server(self):
        """
        Tests the Jupyter notebook option for the generator method.
        """
        jupyter_server = generate_almst_server(self.log_return_dataframe, jupyter=True)
        self.assertIsInstance(jupyter_server, JupyterDash)

    def test_colours_almst_server(self):
        """
        Tests the groups are added correctly, when they are passed using the colours parameter.
        """
        colours_input = {"tech": ['Apple', 'Amazon', 'Facebook', 'Microsoft', 'Netflix', 'Tesla']}
        server_colours = generate_almst_server(self.log_return_dataframe, colours=colours_input)
        for element in server_colours.layout['cytoscape'].elements:
            if len(element) > 1:
                colour_group = element['data']['colour_group']
                self.assertEqual(colour_group, 'tech')

    def test_sizes_almst_server(self):
        """
        Tests the sizes are added correctly when sizes are passed in as a parameter.
        """
        sizes_input = [100, 240, 60, 74, 22, 111]
        server_sizes = generate_almst_server(self.log_return_dataframe, sizes=sizes_input)
        sizes_output = []
        for element in server_sizes.layout['cytoscape'].elements:
            if len(element) > 1:
                size = element['data']['size']
                sizes_output.append(size)
        self.assertEqual(sizes_output, sizes_input)

    def test_default_mst_almst_comparison(self):
        """
        Tests the default, minimal input of the method generate_almst_server.
        """
        default_server = generate_mst_almst_comparison(self.log_return_dataframe)
        self.assertIsInstance(default_server, dash.Dash)
        for element in default_server.layout['cytoscape'].elements:
            if len(element) > 1:
                selectable = element['selectable']
                self.assertEqual(selectable, 'true')

    def test_jupyter_generate_comparison(self):
        """
        Tests the Jupyter notebook option for the generator method.
        """
        jupyter_server = generate_mst_almst_comparison(self.log_return_dataframe, jupyter=True)
        self.assertIsInstance(jupyter_server, JupyterDash)

    def test_default_generate_pmfg_server(self):
        """
        Tests the default, minimal input of the method generate_mst_server.
        """
        default_server = generate_pmfg_server(self.log_return_dataframe)
        self.assertIsInstance(default_server, dash.Dash)

        dist_elements = []

        for element in default_server.layout['cytoscape'].elements:
            if len(element) > 2:
                colour_group = element['data']['colour_group']
                self.assertEqual(colour_group, 'default')
                size = element['data']['size']
                self.assertEqual(size, 0)
            else:
                dist_elements.append(list(element['data'].values()))

        self.assertEqual(dist_elements[0][2], 0.1674)
        self.assertEqual(dist_elements[1][2], 0.2575)
        self.assertEqual(dist_elements[2][2], 0.2967)
        self.assertEqual(dist_elements[3][2], 0.3896)
        self.assertEqual(dist_elements[4][2], 0.1251)
        self.assertEqual(dist_elements[5][2], 0.1726)

    def test_correlation_generate_pmfg_server(self):
        """
        Tests for correlation input option.
        """
        correlation_server = generate_pmfg_server(self.log_return_dataframe, input_type='correlation')
        self.assertIsInstance(correlation_server, dash.Dash)

        corr_elements = []

        for element in correlation_server.layout['cytoscape'].elements:
            if len(element) > 2:
                colour_group = element['data']['colour_group']
                self.assertEqual(colour_group, 'default')
                size = element['data']['size']
                self.assertEqual(size, 0)
            else:
                corr_elements.append(list(element['data'].values()))

        self.assertEqual(corr_elements[0][2], 0.9439)
        self.assertEqual(corr_elements[1][2], 0.8674)
        self.assertEqual(corr_elements[2][2], 0.8239)
        self.assertEqual(corr_elements[3][2], 0.6965)
        self.assertEqual(corr_elements[4][2], 0.9687)
        self.assertEqual(corr_elements[5][2], 0.9404)

    def test_error_generate_pmfg_server(self):
        """
        Tests for error when using unsupported input_type.
        """

        self.assertRaises(ValueError, generate_pmfg_server, log_returns_df=self.log_return_dataframe,
                          input_type='unsupported')

    def test_jupyter_generate_pmfg_server(self):
        """
        Tests the Jupyter notebook option for the generator method.
        """
        jupyter_server = generate_pmfg_server(self.log_return_dataframe, jupyter=True)
        self.assertIsInstance(jupyter_server, JupyterDash)

    def test_colours_pmfg_server(self):
        """
        Tests the groups are added correctly, when they are passed using the colours parameter.
        """
        colours_input = {"tech": ['Apple', 'Amazon', 'Facebook', 'Microsoft', 'Netflix', 'Tesla']}
        server_colours = generate_pmfg_server(self.log_return_dataframe, colours=colours_input)
        for element in server_colours.layout['cytoscape'].elements:
            if len(element) > 2:
                colour_group = element['data']['colour_group']
                self.assertEqual(colour_group, 'tech')

    def test_sizes_pmfg_server(self):
        """
        Tests the sizes are added correctly when sizes are passed in as a parameter.
        """
        sizes_input = [100, 240, 60, 74, 22, 111]
        server_sizes = generate_pmfg_server(self.log_return_dataframe, sizes=sizes_input)
        sizes_output = []
        for element in server_sizes.layout['cytoscape'].elements:
            if len(element) > 2:
                size = element['data']['size']
                sizes_output.append(size)
        self.assertEqual(sizes_output, sizes_input)

    def test_generate_central_peripheral_ranking(self):
        """
        Tests for the expected ranking output for PMFG graph.
        """
        matrix = create_input_matrix(self.log_return_dataframe, distance_matrix_type='angular')
        pmfg = PMFG(matrix, matrix_type='distance')
        pmfg_graph = pmfg.get_graph()
        ranking = generate_central_peripheral_ranking(pmfg_graph)
        expected_ranking = [(5.183389641918858, 'Tesla'), (5.304986512754629, 'Facebook'),
                            (7.3097064737370285, 'Netflix'), (7.553995160955896, 'Apple'),
                            (10.756793026560407, 'Amazon'), (10.868412164786022, 'Microsoft')]
        self.assertEqual(expected_ranking, ranking)
