# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
These methods allows the user to easily deploy graph visualisations given an input file dataframe.
"""

import warnings
import networkx as nx

from mlfinlab.networks.dash_graph import DashGraph, PMFGDash
from mlfinlab.networks.dual_dash_graph import DualDashGraph
from mlfinlab.networks.mst import MST
from mlfinlab.networks.almst import ALMST
from mlfinlab.networks.pmfg import PMFG
from mlfinlab.codependence import get_distance_matrix


def generate_mst_server(log_returns_df, mst_algorithm='kruskal', distance_matrix_type='angular',
                        jupyter=False, colours=None, sizes=None):
    """
    This method returns a Dash server ready to be run.

    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns
        with stock names as columns.
    :param mst_algorithm: (str) A valid MST type such as 'kruskal', 'prim', or 'boruvka'.
    :param distance_matrix_type: (str) A valid sub type of a distance matrix,
        namely 'angular', 'abs_angular', 'squared_angular'.
    :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.
    :param colours: (Dict) A dictionary of key string for category name and value of a list of indexes
        corresponding to the node indexes inputted in the initial dataframe.
    :param sizes: (List) A list of numbers, where the positions correspond to the node indexes inputted
        in the initial dataframe.
    :return: (Dash) Returns the Dash app object, which can be run using run_server.
        Returns a Jupyter Dash object if the parameter jupyter is set to True.
    """
    distance_matrix = create_input_matrix(log_returns_df, distance_matrix_type)

    # Create MST object
    graph = MST(distance_matrix, 'distance', mst_algorithm)

    # If colours are inputted, call set the node colours
    if colours:
        graph.set_node_groups(colours)

    # If sizes are inputted, set the node sizes
    if sizes:
        graph.set_node_size(sizes)

    # If Jupyter is true, create a Jupyter compatible DashGraph object
    if jupyter:
        dash_graph = DashGraph(graph, 'jupyter notebook')
    else:
        dash_graph = DashGraph(graph)

    # Retrieve the server
    server = dash_graph.get_server()

    return server


def create_input_matrix(log_returns_df, distance_matrix_type):
    """
    This method returns the distance matrix ready to be inputted into the Graph class.

    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns
        with stock names as columns.
    :param distance_matrix_type: (str) A valid sub type of a distance matrix,
        namely 'angular', 'abs_angular', 'squared_angular'.
    :return: (pd.Dataframe) A dataframe of a distance matrix.
    """
    # Create correlation matrix
    correlation_matrix = log_returns_df.corr(method='pearson')

    # Valid distance matrix sub types
    valid_matrix_sub_types = ['angular', 'abs_angular', 'squared_angular']

    # If an invalid distance matrix type is used, raise an Error to notify the user
    if distance_matrix_type not in valid_matrix_sub_types:
        msg = "{} is not a valid choice distance matrix sub type. " \
              "Please select one shown in the docstring.".format(distance_matrix_type)
        raise ValueError(msg)

    # Create distance matrix
    distance_matrix = get_distance_matrix(correlation_matrix, distance_metric=distance_matrix_type)

    return distance_matrix


def generate_almst_server(log_returns_df, distance_matrix_type='angular',
                          jupyter=False, colours=None, sizes=None):
    """
    This method returns a Dash server ready to be run.

    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns
        with stock names as columns.
    :param distance_matrix_type: (str) A valid sub type of a distance matrix,
        namely 'angular', 'abs_angular', 'squared_angular'.
    :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.
    :param colours: (Dict) A dictionary of key string for category name and value of a list of indexes
        corresponding to the node indexes inputted in the initial dataframe.
    :param sizes: (List) A list of numbers, where the positions correspond to the node indexes inputted
        in the initial dataframe.
    :return: (Dash) Returns the Dash app object, which can be run using run_server.
        Returns a Jupyter Dash object if the parameter jupyter is set to True.
    """
    distance_matrix = create_input_matrix(log_returns_df, distance_matrix_type)

    # Create ALMST object
    graph = ALMST(distance_matrix, 'distance')

    # If colours are inputted, call set the node colours
    if colours:
        graph.set_node_groups(colours)

    # If sizes are inputted, set the node sizes
    if sizes:
        graph.set_node_size(sizes)

    # If Jupyter is true, create a Jupyter compatible DashGraph object
    if jupyter:
        dash_graph = DashGraph(graph, 'jupyter notebook')
    else:
        dash_graph = DashGraph(graph)

    # Retrieve the server
    server = dash_graph.get_server()

    return server


def generate_mst_almst_comparison(log_returns_df, distance_matrix_type='angular', jupyter=False):
    """
    This method returns a Dash server ready to be run.

    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns
        with stock names as columns.
    :param distance_matrix_type: (str) A valid sub type of a distance matrix,
        namely 'angular', 'abs_angular', 'squared_angular'.
    :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.
    :return: (Dash) Returns the Dash app object, which can be run using run_server.
        Returns a Jupyter Dash object if the parameter jupyter is set to True.
    """
    distance_matrix = create_input_matrix(log_returns_df, distance_matrix_type)

    # Create ALMST and MST objects
    almst = ALMST(distance_matrix, 'distance')
    mst = MST(distance_matrix, 'distance')

    # If Jupyter is true, create a Jupyter compatible DashGraph object
    if jupyter:
        dash_graph = DualDashGraph(almst, mst, 'jupyter notebook')
    else:
        dash_graph = DualDashGraph(almst, mst)

    # Retrieve the server
    server = dash_graph.get_server()

    return server


def generate_pmfg_server(log_returns_df, input_type='distance',
                         jupyter=False, colours=None, sizes=None):
    """
      This method returns a PMFGDash server ready to be run.

      :param log_returns_df: (pd.Dataframe) An input dataframe of log returns
          with stock names as columns.
      :param input_type: (str) A valid input type correlation or distance. Inputting correlation will add the edges
          by largest to smallest, instead of smallest to largest.
      :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.
      :param colours: (Dict) A dictionary of key string for category name and value of a list of indexes
          corresponding to the node indexes inputted in the initial dataframe.
      :param sizes: (List) A list of numbers, where the positions correspond to the node indexes inputted
          in the initial dataframe.
      :return: (Dash) Returns the Dash app object, which can be run using run_server.
          Returns a Jupyter Dash object if the parameter jupyter is set to True.
      """
    if input_type == 'distance':
        distance_matrix = create_input_matrix(log_returns_df, 'angular')
        graph = PMFG(distance_matrix, 'distance')
    elif input_type == 'correlation':
        correlation_matrix = log_returns_df.corr(method='pearson')
        graph = PMFG(correlation_matrix, 'correlation')
    else:
        # If an invalid input matrix type is used, raise an Error to notify the user
        msg = "{} is not a valid choice input type. " \
              "Please select correlation or distance.".format(input_type)
        raise ValueError(msg)

    # If colours are inputted, call set the node colours
    if colours:
        graph.set_node_groups(colours)

    # If sizes are inputted, set the node sizes
    if sizes:
        graph.set_node_size(sizes)

    # If Jupyter is true, create a Jupyter compatible DashGraph object
    if jupyter:
        dash_graph = PMFGDash(graph, 'jupyter notebook')
    else:
        dash_graph = PMFGDash(graph)

    # Retrieve the server
    server = dash_graph.get_server()

    return server


def generate_central_peripheral_ranking(nx_graph):
    """
    Given a NetworkX graph, this method generates and returns a ranking of centrality.
    The input should be a distance based PMFG.

    The ranking combines multiple centrality measures to calculate an overall ranking of how central or peripheral the
    nodes are.
    The smaller the ranking, the more peripheral the node is. The larger the ranking, the more central the node is.

    The factors contributing to the ranking include Degree, Eccentricity, Closeness Centrality, Second Order Centrality,
    Eigen Vector Centrality and Betweenness Centrality. The formula for these measures can be found on the NetworkX
    documentation (https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html)

    :param nx_graph: (nx.Graph) NetworkX graph object. You can call get_graph() on the MST, ALMST and PMFG to retrieve
        the nx.Graph.
    :return: (List) Returns a list of tuples of ranking value to node.
    """

    # Weighted and unweighted degree measure of the graph
    degrees = nx.degree(nx_graph, weight='weight')
    degrees_unweighted = nx.degree(nx_graph)

    # Eccentricity of the graph nodes
    eccentricity = nx.eccentricity(nx_graph)

    # Closeness centrality of weighted and unweighted graphs
    closeness = nx.closeness_centrality(nx_graph, distance='weight')
    closeness_unweighted = nx.closeness_centrality(nx_graph)

    # Second order centrality rating
    with warnings.catch_warnings():  # Silencing specific PendingDeprecationWarning
        warnings.filterwarnings('ignore', r'the matrix subclass is not the recommended way')
        second_order = nx.second_order_centrality(nx_graph)

    # Eigen vector centrality for unweighted graph.
    eigen_vector_centrality_unweighted = nx.eigenvector_centrality(nx_graph)

    # Betweenness Centrality for both weighted and unweighted graph.
    betweenness = nx.betweenness_centrality(nx_graph, weight='weight')
    betweenness_unweighted = nx.betweenness_centrality(nx_graph)

    ranked_nodes = []
    for node in list(nx_graph.nodes()):
        # Degrees, Betweenness, Eigenvector Centrality and Closeness produce larger values for more central nodes.
        ranking = 0
        ranking += degrees[node] + degrees_unweighted[node]
        ranking += betweenness[node] + betweenness_unweighted[node]
        ranking += eigen_vector_centrality_unweighted[node]
        ranking += closeness[node] + closeness_unweighted[node]

        # Second Order Centrality, Eccentricity produce smaller values for more central nodes.
        # Second order centrality is divided to normalise its impact on the ranking.
        ranking += (second_order[node] / 100) * -1
        ranking += eccentricity[node] * -1

        ranked_nodes.append((ranking, node))

    ranked_nodes.sort(key=lambda tup: tup[0])

    return ranked_nodes
