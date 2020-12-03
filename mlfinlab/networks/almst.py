# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This file contains Graph classes, which create NetworkX's Graph objects from matrices.
"""

import heapq
import itertools
from itertools import count

import networkx as nx
import numpy as np
import pandas as pd
from mlfinlab.networks.graph import Graph


class ALMST(Graph):
    """
    ALMST is a subclass of Graph which creates a ALMST Graph object.
    The ALMST class converts a distance matrix input into a ALMST matrix. This is then used to create a nx.Graph object.
    """

    def __init__(self, matrix, matrix_type, mst_algorithm='kruskal'):
        """
        Initialises the ALMST and sets the self.graph attribute as the ALMST graph.

        :param matrix: (pd.Dataframe) Input matrices such as a distance or correlation matrix.
        :param matrix_type: (str) Name of the matrix type (e.g. "distance" or "correlation").
        :param mst_algorithm: (str) Valid MST algorithm types include 'kruskal', 'prim'.
            By default, MST algorithm uses Kruskal's.
        """
        super().__init__(matrix_type)

        # Create ALMST matrix
        if mst_algorithm == 'kruskal':
            almst_matrix = self.create_almst_kruskals(matrix)
        elif mst_algorithm == 'prim':
            almst_matrix = self.create_almst(matrix)
        else:
            msg = "{} is not a valid MST algorithm type. " \
                  "Please select one shown in the docstring.".format(mst_algorithm)
            raise ValueError(msg)
        # Create a nx.Graph object from ALSMT dataframe
        self.graph = nx.from_pandas_adjacency(almst_matrix)
        # Create positions
        self.pos = nx.spring_layout(self.graph)

    @staticmethod
    def create_almst_kruskals(matrix):
        """
        This method converts the input matrix into a ALMST matrix.

        ! Currently only works with distance input matrix

        :param matrix: (pd.Dataframe) Input matrix.
        :return: (pd.Dataframe) ALMST matrix with all other edges as 0 values.
        """

        # Create an empty matrix for output
        a_mat = np.zeros(shape=(len(matrix), len(matrix)))
        almst_matrix = pd.DataFrame(a_mat, columns=list(matrix), index=list(matrix))

        # Create a list of clusters
        # Initially, every node is initialised as its own cluster
        clusters = []
        for i in range(len(matrix.columns)):
            clusters.append([i])

        num_edges = 0

        # Until the number of edges equals the number of nodes -1, keep adding edges
        while num_edges < len(matrix.columns) - 1:

            # Generate ordered heap of clusters
            heap = ALMST._generate_ordered_heap(matrix, clusters)

            # Pop the minimum average distance clusters from the heap
            _, _, edge = heapq.heappop(heap)

            # Add the edge to the matrix
            edge_weight = matrix.iat[edge['min_i'], edge['min_j']]
            almst_matrix.iat[edge['min_i'], edge['min_j']] = edge_weight
            almst_matrix.iat[edge['min_j'], edge['min_i']] = edge_weight

            # Merge the clusters containing the new edge together
            cluster_1 = clusters[edge['cx']]
            cluster_2 = clusters[edge['cy']]
            clusters.pop(edge['cy'])
            clusters.pop(edge['cx'])
            clusters.append(cluster_1+cluster_2)

            num_edges += 1

        return almst_matrix

    @staticmethod
    def _generate_ordered_heap(matrix, clusters):
        """
        Given the matrix of edges, and the list of clusters, generate a heap ordered by the average distance between the clusters.

        :param matrix: (pd.Dataframe) Input matrix of the distance matrix.
        :param clusters: (List) A list of clusters, where each list contains a list of nodes within the cluster.
        :return: (Heap) Returns a heap ordered by the average distance between the clusters.
        """

        cnt = count()
        heap = []

        # Generate pairwise combination between clusters
        pairwise_combinations = list(itertools.combinations(range(len(clusters)), 2))

        # For cluster 1 and cluster 2 in the pairwise combination lists
        for c_x, c_y in pairwise_combinations:
            # Calculate the average distance between the 2 clusters
            # And keep track of the minimum distance edge which will link the two clusters
            average, min_i, min_j = ALMST._calculate_average_distance(matrix, clusters, c_x, c_y)
            # Push the edge to the heap
            heapq.heappush(heap, (average, next(cnt), {'min_i': min_i, 'min_j': min_j, 'cx': c_x, 'cy': c_y}))

        return heap

    @staticmethod
    def _calculate_average_distance(matrix, clusters, c_x, c_y):
        """
        Given two clusters, calculates the average distance between the two.

        :param matrix: (pd.Dataframe) Input matrix with all edges.
        :param clusters: (List) List of clusters.
        :param c_x: (int) Cluster x, where x is the index of the cluster.
        :param c_y: (int) Cluster y, where y is the index of the cluster.
        """

        # If both of the clusters only contain one node, the average edge is the edge between the nodes.
        if len(clusters[c_x]) == 1 and len(clusters[c_y]) == 1:
            average = matrix.iat[clusters[c_x][0], clusters[c_y][0]]
            result = average, clusters[c_x][0], clusters[c_y][0]

        # If the cluster one is a single element cluster, but cluster two contains many elements
        elif len(clusters[c_x]) == 1:
            c_one_element = clusters[c_x][0]
            result = ALMST._get_min_edge(c_one_element, clusters[c_y], matrix)

        # If cluster y is a single element cluster, but cluster x is not
        elif len(clusters[c_y]) == 1:
            c_one_element = clusters[c_y][0]
            result = ALMST._get_min_edge(c_one_element, clusters[c_x], matrix)

        # Else, both of the clusters are multi-element
        else:
            # For all nodes in c_x and c_y, compute the average distance between each pair
            result = ALMST._get_min_edge_clusters(clusters[c_x], clusters[c_y], matrix)

        # Returns the average, and the minimum edge merging the two clusters
        return result

    @staticmethod
    def _get_min_edge(node, cluster, matrix):
        """
        Returns the minimum edge tuple given a node and a cluster.

        :param node: (str) String of the node name.
        :param cluster: (list) List of node names.
        :param matrix: (pd.DataFrame) A matrix of all edges.
        :return: (tuple) A tuple of average distance from node to the cluster, and the minimum edge nodes, i and j.
        """
        min_edge_and_node = (np.inf, 0, 0)
        summ = 0
        for element in cluster:
            edge = matrix.iat[element, node]
            if edge < min_edge_and_node[0]:
                min_edge_and_node = (edge, element, node)
            summ += edge
        average = summ / len(cluster)
        return average, min_edge_and_node[1], min_edge_and_node[2]

    @staticmethod
    def _get_min_edge_clusters(cluster_one, cluster_two, matrix):
        """
        Returns a tuple of the minimum edge and the average length for two clusters.

        :param cluster_one: (list) List of node names.
        :param cluster_two: (list) List of node names.
        :param matrix: (pd.DataFrame) A matrix of all edges.
        :return: (tuple) A tuple of average distance between the clusters, and the minimum edge nodes, i and j.
        """
        min_edge_and_node = (np.inf, 0, 0)
        summ = 0
        average = 0
        for node_i in cluster_one:
            for node_j in cluster_two:
                edge = matrix.iat[node_i, node_j]
                if edge < min_edge_and_node[0]:
                    min_edge_and_node = (edge, node_i, node_j)
                summ += edge
        total_length = len(cluster_one) + len(cluster_two)
        average += summ / total_length
        return average, min_edge_and_node[1], min_edge_and_node[2]

    @staticmethod
    def create_almst(matrix):
        """
        Creates and returns a ALMST given an input matrix using Prim's algorithm.

        :param matrix: (pd.Dataframe) Input distance matrix of all edges.
        :return: (pd.Dataframe) Returns the ALMST in matrix format.
        """

        # Create an empty ALMST matrix
        a_mat = np.zeros(shape=(len(matrix), len(matrix)))
        almst_matrix = pd.DataFrame(a_mat, columns=list(matrix), index=list(matrix))

        # Create a set of visited nodes, and visit the first node
        visited = set()
        visited.add(0)

        # Create a set of children nodes (all nodes connected to any visited nodes)
        children = set()

        # While not all the nodes have been visited keep adding edges
        while len(visited) < len(matrix):
            visited, children, almst_matrix = ALMST._add_next_edge(visited, children, matrix, almst_matrix)

        # Returns the df of the ALMST matrix
        return almst_matrix

    @staticmethod
    def _add_next_edge(visited, children, matrix, almst_matrix):
        """
        Adds the next edge with the minimum average distance.

        :param visited: (Set) A set of visited nodes.
        :param children: (Set) A set of children or frontier nodes, to be visited.
        :param matrix: (pd.Dataframe) Input distance matrix of all edges.
        :param almst_matrix: (pd.Dataframe) The ALMST matrix.

        :return: (Tuple) Returns the sets visited and children, and the matrix almst_matrix.
        """

        cnt = count()

        # For all visited nodes, add any nodes connected to the visited node
        for visited_node in visited:
            for i, _ in enumerate(matrix.iloc[visited_node]):
                if i not in children and i != visited_node:
                    children.add(i)

        # Remove any already visited nodes from the list of children
        children = children - visited

        # Heap of children nodes
        heap = []

        for child in children:
            sum_distance = 0
            min_edge_and_node = (np.inf, np.inf)

            # For every child, calculate the average distance to the cluster of visited nodes
            for visited_node in visited:
                distance = matrix.iat[visited_node, child]
                sum_distance += distance

                # Keep track of the minimum edge to connect child to the cluster of visited nodes
                if distance < min_edge_and_node[0]:
                    min_edge_and_node = (distance, visited_node)

            # Calculate the average distance and add it to the heap
            average_distance = sum_distance / len(visited)
            heapq.heappush(heap, (average_distance, next(cnt), {'child': child, 'min_edge_node': min_edge_and_node}))

        # Pop the child with the minimum average distance
        _, _, child_to_add = heapq.heappop(heap)

        # This child becomes the next visited node
        add_node = child_to_add['child']

        # This is the node already visited, to connect with the child node
        min_edge_node = child_to_add['min_edge_node'][1]

        # Add the edge child to min_edge_node
        almst_matrix.iat[add_node, min_edge_node] = child_to_add['min_edge_node'][0]
        almst_matrix.iat[min_edge_node, add_node] = child_to_add['min_edge_node'][0]

        # Visit the child
        visited.add(add_node)

        # Return the modified sets and matrix
        return visited, children, almst_matrix
