# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Tests the Entropy Pooling algorithm proposed by Meucci.
"""

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.bayesian import EntropyPooling
from mlfinlab.portfolio_optimization.estimators import ReturnsEstimators


class TestEntropyPooling(unittest.TestCase):
    # pylint: disable=too-many-public-methods
    """
    Tests different functions of the EntropyPooling class.
    """

    def setUp(self):
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_entropy_pooling_for_all_constraints(self):
        """
        Test the Entropy Pooling algorithm for both equality and inequality constraints.
        """

        returns = ReturnsEstimators().calculate_returns(self.data.iloc[:, :5])
        num_time_stamps = returns.shape[0]
        p_initial = np.array([1 / num_time_stamps] * num_time_stamps)
        equality_matrix = returns[['EEM', 'EWG']].values
        equality_vector = [2.3e-02, 3.3e-02]
        inequality_matrix = returns[['TIP', 'EWJ', 'EFA']].values
        inequality_vector = [1.1e-3, 2.2e-4, 3.8e-6]
        ep_solver = EntropyPooling()
        ep_solver.calculate_posterior_probabilities(prior_probabilities=p_initial,
                                                    equality_matrix=equality_matrix,
                                                    equality_vector=equality_vector,
                                                    inequality_matrix=inequality_matrix,
                                                    inequality_vector=inequality_vector)
        assert len(ep_solver.posterior_probabilities) == len(p_initial)
        np.testing.assert_almost_equal(equality_matrix.T.dot(ep_solver.posterior_probabilities)[0][0].round(3), equality_vector[0])
        np.testing.assert_almost_equal(equality_matrix.T.dot(ep_solver.posterior_probabilities)[1][0].round(3), equality_vector[1])
        assert ep_solver.posterior_probabilities.T.dot(inequality_matrix)[0][0] >= 1.1e-3
        assert inequality_matrix.T.dot(ep_solver.posterior_probabilities)[1][0] >= 2.2e-4
        assert inequality_matrix.T.dot(ep_solver.posterior_probabilities)[2][0] >= 3.8e-6

    def test_entropy_pooling_for_leq_inequality_constraints(self):
        #pylint: disable=invalid-name
        """
        Test the Entropy Pooling algorithm for leq inequality constraints.
        """

        returns = ReturnsEstimators().calculate_returns(self.data.iloc[:, :5])
        num_time_stamps = returns.shape[0]
        p_initial = np.array([1 / num_time_stamps] * num_time_stamps)
        equality_matrix = returns[['EEM', 'EWG']].values
        equality_vector = [2.3e-02, 3.3e-02]
        inequality_matrix = returns[['TIP', 'EWJ', 'EFA']].values
        inequality_matrix[:, 1] *= -1
        inequality_vector = [1.1e-3, -2.2e-4, 3.8e-6]
        ep_solver = EntropyPooling()
        ep_solver.calculate_posterior_probabilities(prior_probabilities=p_initial,
                                                    equality_matrix=equality_matrix,
                                                    equality_vector=equality_vector,
                                                    inequality_matrix=inequality_matrix,
                                                    inequality_vector=inequality_vector)
        assert len(ep_solver.posterior_probabilities) == len(p_initial)
        np.testing.assert_almost_equal(equality_matrix.T.dot(ep_solver.posterior_probabilities)[0][0].round(3), equality_vector[0])
        np.testing.assert_almost_equal(equality_matrix.T.dot(ep_solver.posterior_probabilities)[1][0].round(3), equality_vector[1])
        assert ep_solver.posterior_probabilities.T.dot(inequality_matrix)[0][0] >= 1.1e-3
        assert inequality_matrix.T.dot(ep_solver.posterior_probabilities)[1][0] <= 2.2e-4
        assert inequality_matrix.T.dot(ep_solver.posterior_probabilities)[2][0] >= 3.8e-6

    def test_entropy_pooling_for_only_equality_constraints(self):
        # pylint: disable=invalid-name
        """
        Test Entropy Pooling when passing only equality constraints.
        """

        returns = ReturnsEstimators().calculate_returns(self.data.iloc[:, :5])
        num_time_stamps = returns.shape[0]
        p_initial = np.array([1 / num_time_stamps] * num_time_stamps)
        equality_matrix = returns[['EEM', 'EWG']].values
        equality_vector = [2.3e-02, 3.3e-02]
        ep_solver = EntropyPooling()
        ep_solver.calculate_posterior_probabilities(prior_probabilities=p_initial,
                                                    equality_matrix=equality_matrix,
                                                    equality_vector=equality_vector)
        assert len(ep_solver.posterior_probabilities) == len(p_initial)
        np.testing.assert_almost_equal(equality_matrix.T.dot(ep_solver.posterior_probabilities)[0][0].round(3),
                                       equality_vector[0])
        np.testing.assert_almost_equal(equality_matrix.T.dot(ep_solver.posterior_probabilities)[1][0].round(3),
                                       equality_vector[1])

    def test_value_error_prob_sum(self):
        """
        Test ValueError when the prior probabilities do not sum to 1.
        """

        with self.assertRaises(ValueError):
            returns = ReturnsEstimators().calculate_returns(self.data.iloc[:, :5])
            num_time_stamps = returns.shape[0]
            p_initial = np.array([1 / 2] * num_time_stamps)
            equality_matrix = returns[['EEM', 'EWG']].values
            equality_vector = [2.3e-02, 3.3e-02]
            inequality_matrix = returns[['TIP', 'EWJ', 'EFA']].values
            inequality_vector = [1.1e-3, 2.2e-4, 3.8e-6]
            ep_solver = EntropyPooling()
            ep_solver.calculate_posterior_probabilities(prior_probabilities=p_initial,
                                                        equality_matrix=equality_matrix,
                                                        equality_vector=equality_vector,
                                                        inequality_matrix=inequality_matrix,
                                                        inequality_vector=inequality_vector)

    def test_value_error_all_inputs_null(self):
        """
        Test ValueError when no equality and inequality matricies have been specified.
        """

        with self.assertRaises(ValueError):
            returns = ReturnsEstimators().calculate_returns(self.data.iloc[:, :5])
            num_time_stamps = returns.shape[0]
            p_initial = np.array([1 / num_time_stamps] * num_time_stamps)
            equality_vector = [2.3e-02, 3.3e-02]
            inequality_vector = [1.1e-3, 2.2e-4, 3.8e-6]
            ep_solver = EntropyPooling()
            ep_solver.calculate_posterior_probabilities(prior_probabilities=p_initial,
                                                        equality_vector=equality_vector,
                                                        inequality_vector=inequality_vector)

    def test_value_error_for_no_equality_vector(self):
        """
        Test ValueError when no equality vector is specified.
        """

        with self.assertRaises(ValueError):
            returns = ReturnsEstimators().calculate_returns(self.data.iloc[:, :5])
            num_time_stamps = returns.shape[0]
            p_initial = np.array([1 / num_time_stamps] * num_time_stamps)
            equality_matrix = returns[['EEM', 'EWG']].values
            ep_solver = EntropyPooling()
            ep_solver.calculate_posterior_probabilities(prior_probabilities=p_initial,
                                                        equality_matrix=equality_matrix,
                                                        equality_vector=[])

    def test_value_error_for_no_inequality_vector(self):
        """
        Test ValueError when no inequality vector is specified.
        """

        with self.assertRaises(ValueError):
            returns = ReturnsEstimators().calculate_returns(self.data.iloc[:, :5])
            num_time_stamps = returns.shape[0]
            p_initial = np.array([1 / num_time_stamps] * num_time_stamps)
            inequality_matrix = returns[['TIP', 'EWJ', 'EFA']].values
            ep_solver = EntropyPooling()
            ep_solver.calculate_posterior_probabilities(prior_probabilities=p_initial,
                                                        inequality_matrix=inequality_matrix)

    def test_value_error_for_diff_lengths_of_equality_constraints(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when the lengths of equality matrix and equality vector are different.
        """

        with self.assertRaises(ValueError):
            returns = ReturnsEstimators().calculate_returns(self.data.iloc[:, :5])
            num_time_stamps = returns.shape[0]
            p_initial = np.array([1 / num_time_stamps] * num_time_stamps)
            equality_matrix = returns[['EEM', 'EWG']].values
            equality_vector = [2.3e-02]
            ep_solver = EntropyPooling()
            ep_solver.calculate_posterior_probabilities(prior_probabilities=p_initial,
                                                        equality_matrix=equality_matrix,
                                                        equality_vector=equality_vector)

    def test_value_error_for_diff_lengths_of_inequality_constraints(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when the lengths of inequality matrix and inequality vector are different.
        """

        with self.assertRaises(ValueError):
            returns = ReturnsEstimators().calculate_returns(self.data.iloc[:, :5])
            num_time_stamps = returns.shape[0]
            p_initial = np.array([1 / num_time_stamps] * num_time_stamps)
            inequality_matrix = returns[['EEM', 'EWG']].values
            inequality_vector = [2.3e-02]
            ep_solver = EntropyPooling()
            ep_solver.calculate_posterior_probabilities(prior_probabilities=p_initial,
                                                        inequality_matrix=inequality_matrix,
                                                        inequality_vector=inequality_vector)

    def test_plotting_of_histogram(self):
        """
        Test the histogram generated from new probabilities.
        """

        returns = ReturnsEstimators().calculate_returns(self.data.iloc[:, :5])
        num_time_stamps = returns.shape[0]
        p_initial = np.array([1 / num_time_stamps] * num_time_stamps)
        equality_matrix = returns[['EEM']].values
        equality_vector = [2.3e-02]
        ep_solver = EntropyPooling()
        ep_solver.calculate_posterior_probabilities(prior_probabilities=p_initial,
                                                    equality_matrix=equality_matrix,
                                                    equality_vector=equality_vector)
        histogram = ep_solver.generate_histogram(historical_market_vector=returns[['EEM']], num_bins=20)
        assert len(histogram.get_children()) == 21
        rectangle_1 = histogram.get_children()[0]
        rectangle_2 = histogram.get_children()[1]
        assert rectangle_1.get_width() > 0
        assert rectangle_1.get_width() == rectangle_2.get_width()
