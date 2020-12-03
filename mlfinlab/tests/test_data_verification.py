# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Tests the data verification methods for synthetic data in
data_generation/data_verification.py.
"""
import unittest
import os
import numpy as np
import matplotlib
import mlfinlab.data_generation.data_verification as data_verification
from mlfinlab.data_generation.data_verification import plot_time_series_dependencies
from mlfinlab.data_generation.correlated_random_walks import generate_cluster_time_series
from mlfinlab.data_generation.data_verification import (
    plot_pairwise_dist,
    plot_eigenvalues,
    plot_eigenvectors,
    plot_hierarchical_structure,
    plot_mst_degree_count,
    plot_stylized_facts,
    plot_optimal_hierarchical_cluster
)


class TestDataVerificationMethods(unittest.TestCase):
    """
    Test the data verification methods for synthetic data.
    """

    def setUp(self):
        """
        Sets random number generator seeds and
        file path for the corrgan generator model.
        """

        np.random.seed(2814)
        project_path = os.path.dirname(__file__)
        path = project_path + "/test_data"
        self.generator_path = path

    def test_correlated_random_walks_plot(self):
        """
        Tests that correlated random walks dependence plots are plotted normally.
        """

        # Needed to avoid actual image plotting.
        matplotlib.use("Template")

        n_series = 50
        t_samples = 1200
        k_clusters = 1
        d_clusters = 4
        data_series = generate_cluster_time_series(n_series, t_samples, k_clusters, d_clusters)
        self.assertFalse(plot_time_series_dependencies(data_series) is None)

        matplotlib.pyplot.show()

    def test_compute_eigenvalues(self):
        # pylint: disable=protected-access
        """
        Test that the computed eigenvalues are valid.
        """

        eigenvalues = data_verification._compute_eigenvalues([np.diag((1, 2, 3))])
        self.assertTrue(1 in eigenvalues)
        self.assertTrue(2 in eigenvalues)
        self.assertTrue(3 in eigenvalues)

    def test_compute_pf_vec(self):
        # pylint: disable=protected-access
        """
        Test that the computed Perron-Frobenius vector is valid.

        Example PF vector reference:
        https://core.ac.uk/download/pdf/82012819.pdf
        """

        pf_vector = data_verification._compute_pf_vec(
            np.array([[[1, 2, 1], [-0.4, 1, 1], [-0.4, 5, 8]]])
        )
        pf_vector = np.round(pf_vector, 6)
        self.assertTrue(0.161752 in pf_vector)
        self.assertTrue(0.121112 in pf_vector)
        self.assertTrue(0.979371 in pf_vector)

    def test_compute_degree_counts(self):
        # pylint: disable=protected-access
        """
        Test that the computed degree count of MSTs are valid.

        Example MST reference:
        https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
        """
        mst_mat = np.array(
            [[[0, 2, 0, 6, 0], [2, 0, 3, 8, 5], [0, 3, 0, 0, 7], [6, 8, 0, 0, 9], [0, 5, 7, 9, 0]]]
        )
        deg_count = data_verification._compute_degree_counts(mst_mat)
        self.assertTrue(np.sum(deg_count) == 1.25)

    def test_all_plots(self):
        """
        Tests that all plots are plotted normally.
        """

        # Needed to avoid actual image plotting.
        matplotlib.use("Template")

        # Some dummy data for testing purposes only.
        gen_dummy = np.array([[[1, 0.45518, 0.439411],
                               [0.45518, 1, 0.703234],
                               [0.439411, 0.703234, 1]]])

        emp_dummy = np.array([[[1, 0.51727604, 0.64076545],
                               [0.51727604, 1, 0.67529106],
                               [0.64076545, 0.67529106, 1]]])

        # Need to close plots mid way in the test to avoid too many plots warning.
        self.assertFalse(plot_pairwise_dist(emp_dummy, gen_dummy) is None)
        self.assertFalse(plot_eigenvalues(emp_dummy, gen_dummy) is None)
        self.assertFalse(plot_eigenvectors(emp_dummy, gen_dummy) is None)
        self.assertFalse(plot_hierarchical_structure(emp_dummy, gen_dummy) is None)
        self.assertFalse(plot_mst_degree_count(emp_dummy, gen_dummy) is None)
        self.assertFalse(plot_optimal_hierarchical_cluster(emp_dummy[0]) is None)
        plot_stylized_facts(emp_dummy, gen_dummy)
