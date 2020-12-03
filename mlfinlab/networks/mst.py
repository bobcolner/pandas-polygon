# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This file contains Graph classes, which create NetworkX's Graph objects from matrices.
"""

import networkx as nx
from mlfinlab.networks.graph import Graph


class MST(Graph):
    """
    MST is a subclass of Graph which creates a MST Graph object.
    """

    def __init__(self, matrix, matrix_type, mst_algorithm='kruskal'):
        """
        Creates a MST Graph object and stores the MST inside graph attribute.

        :param matrix: (pd.Dataframe) Input matrices such as a distance or correlation matrix.
        :param matrix_type: (str) Name of the matrix type (e.g. "distance" or "correlation").
        :param mst_algorithm: (str) Valid MST algorithm types include 'kruskal', 'prim', or 'boruvka'.
            By default, MST algorithm uses Kruskal's.
        """
        super().__init__(matrix_type)
        self.graph = self.create_mst(matrix, mst_algorithm)
        self.pos = nx.spring_layout(self.graph)

    @staticmethod
    def create_mst(matrix, algorithm='kruskal'):
        """
        This method converts the input matrix into a MST graph.

        :param matrix: (pd.Dataframe) Input matrix.
        :param algorithm: (str) Valid MST algorithm types include 'kruskal', 'prim', or 'boruvka'.
            By default, MST algorithm uses Kruskal's.
        """
        valid_algo_types = ['kruskal', 'prim', 'boruvka']
        # If an invalid mst algorithm is used, raise an Error to notify the user
        if algorithm not in valid_algo_types:
            msg = "{} is not a valid MST algorithm type. " \
                  "Please select one shown in the docstring.".format(algorithm)
            raise ValueError(msg)
        return nx.minimum_spanning_tree(nx.Graph(matrix), algorithm=algorithm)
