# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Tests the data verification methods for synthetic data in
data_generation/correlated_random_walks.py.
"""
import unittest
import numpy as np
from mlfinlab.data_generation.correlated_random_walks import generate_cluster_time_series


class TestCorrelatedRandomWalks(unittest.TestCase):
    """
    Test the generated correlated random walks time series.
    """

    def setUp(self):
        """
        Sets random number generator seed.
        """
        np.random.seed(2814)
        self.n_series = [40, 50, 50, 50]
        self.t_samples = [5, 1200, 1200, 1200]
        self.k_clusters = [5, 1, 10, 5]
        self.d_clusters = [2, 4, 1, 2]

    @staticmethod
    def _data_exists(mats):
        """
        Returns whether mats exists and is greater and 0
        """

        return mats is not None and len(mats) > 0

    def test_time_series_returned(self):
        """
        Tests that data generated from all parameters exists for multiple dimensions and
        parameters.
        """

        # Test various parameters.
        for i in range(len(self.n_series)):
            data_series = generate_cluster_time_series(
                self.n_series[i], self.t_samples[i], self.k_clusters[i], self.d_clusters[i]
            )
            self.assertTrue(self._data_exists(data_series))

    def test_time_series_exists(self):
        """
        Tests that data generated from all parameters exists for multiple dimensions and
        parameters.
        """

        # Test various parameters.
        for i in range(len(self.n_series)):
            data_series = generate_cluster_time_series(
                self.n_series[i], self.t_samples[i], self.k_clusters[i], self.d_clusters[i]
            )
            self.assertTrue(data_series.shape == (self.t_samples[i], self.n_series[i]))

    @staticmethod
    def test_time_series_small_data():
        """
        Tests that all parameters return valid data.
        Valid data is defined as not all values equal to 0.
        """

        arr_sample = [
            [98.20787691, 99.10215448, 99.69180009, 99.14313262],
            [98.47864948, 98.1625304, 99.72537914, 98.36703471],
            [99.46151425, 98.56787149, 98.82494457, 97.0846968],
            [97.56282451, 95.81562614, 99.13930959, 96.86162304],
        ]

        data_series = generate_cluster_time_series(
            4, 4, 2, 1, dists_clusters=("student-t", "normal", "student-t", "normal", "student-t")
        )
        np.testing.assert_array_almost_equal(data_series.values, arr_sample, 4)
