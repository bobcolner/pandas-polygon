# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Tests the Vanilla Black Litterman algorithm.
"""

import unittest
import os
import pandas as pd
import numpy as np
from numpy.linalg import inv
from mlfinlab.portfolio_optimization.bayesian import VanillaBlackLitterman


class TestVanillaBlackLitterman(unittest.TestCase):
    # pylint: disable=too-many-public-methods
    """
    Tests different functions of the BlackLitterman algorithm class.
    """

    def setUp(self):
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    @staticmethod
    def test_black_litterman_with_absolute_views():
        """
        Test the weights calculated by the Black Litterman algorithm for absolute investor views.
        """

        tickers = ['A', 'B']
        covariance = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) * 10e-4
        market_cap_weights = [0.44, 0.56]
        views = [0.02, 0.04]
        pick_list = [{'A': 1}, {'B': 1}]

        bl_model = VanillaBlackLitterman()
        bl_model.allocate(covariance=covariance,
                          market_capitalised_weights=market_cap_weights,
                          investor_views=views,
                          pick_list=pick_list,
                          asset_names=tickers)
        weights = bl_model.weights.values[0]
        assert len(weights) == 2
        assert np.round(weights[0], 2) == 0.14
        assert np.round(weights[1], 2) == 0.86
        np.testing.assert_almost_equal(np.sum(weights), 1)

    @staticmethod
    def test_black_litterman_with_relative_views():
        """
        Test the weights calculated by the Black Litterman algorithm for relative investor views.
        """

        tickers = ['A', 'B']
        covariance = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) * 10e-4
        market_cap_weights = [0.44, 0.56]
        views = [0.02]
        pick_list = [{'A': 1, 'B': -1}]

        bl_model = VanillaBlackLitterman()
        bl_model.allocate(covariance=covariance,
                          market_capitalised_weights=market_cap_weights,
                          investor_views=views,
                          pick_list=pick_list)
        weights = bl_model.weights.values[0]
        assert len(weights) == 2
        assert np.round(weights[0], 2) == 0.35
        assert np.round(weights[1], 2) == 0.65
        np.testing.assert_almost_equal(np.sum(weights), 1)

    @staticmethod
    def test_original_black_litterman_results_for_first_view():
        #pylint: disable=invalid-name, protected-access, unsubscriptable-object
        """
        Test and replicate the results of original Black Litterman paper for first view - Germany vs Rest of Europe.
        """

        def calculate_weights(risk_aversion, covariance, expected_returns):
            return (inv(covariance).dot(expected_returns.T)) / risk_aversion

        countries = ['AU', 'CA', 'FR', 'DE', 'JP', 'UK', 'US']

        # Table 1 of the He-Litterman paper: Correlation matrix
        correlation = pd.DataFrame([
            [1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
            [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
            [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
            [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
            [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
            [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
            [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]
        ], index=countries, columns=countries)

        # Table 2 of the He-Litterman paper: Volatilities
        volatilities = pd.DataFrame([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187], index=countries, columns=["vol"])
        covariance = volatilities.dot(volatilities.T) * correlation

        # Table 2 of the He-Litterman paper: Market-capitalised weights
        market_weights = pd.DataFrame([0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615], index=countries, columns=["CapWeight"])

        bl_model = VanillaBlackLitterman()
        equilibrium_returns = bl_model._calculate_implied_equilibrium_returns(2.5, covariance, market_weights)
        equilibrium_returns = (equilibrium_returns * 100).round(1).T
        assert list(equilibrium_returns.values[0]) == [3.9, 6.9, 8.4, 9.0, 4.3, 6.8, 7.6]

        # View-1
        bl_model = VanillaBlackLitterman()
        views = [0.05]
        pick_list = [
            {
                "DE": 1.0,
                "FR": -market_weights.loc["FR"]/(market_weights.loc["FR"] + market_weights.loc["UK"]),
                "UK": -market_weights.loc["UK"] / (market_weights.loc["FR"] + market_weights.loc["UK"])
            }
        ]
        bl_model.allocate(covariance=covariance,
                          market_capitalised_weights=market_weights,
                          investor_views=views,
                          pick_list=pick_list,
                          asset_names=covariance.columns,
                          tau=0.05,
                          risk_aversion=2.5)
        expected_returns = (bl_model.posterior_expected_returns * 100).round(1)
        assert list(expected_returns.values[0]) == [4.3, 7.6, 9.3, 11.0, 4.5, 7.0, 8.1]

        weights = calculate_weights(2.5, bl_model.posterior_covariance, bl_model.posterior_expected_returns)
        weights = (weights * 100).round(1).T
        assert list(weights[0]) == [1.5, 2.1, -4.0, 35.4, 11., -9.5, 58.6]

    @staticmethod
    def test_original_black_litterman_results_for_second_view():
        # pylint: disable=invalid-name, unsubscriptable-object
        """
        Test and replicate the results of original Black Litterman paper for second view - Canada vs US
        """

        def calculate_weights(risk_aversion, covariance, expected_returns):
            return (inv(covariance).dot(expected_returns.T)) / risk_aversion

        countries = ['AU', 'CA', 'FR', 'DE', 'JP', 'UK', 'US']

        # Table 1 of the He-Litterman paper: Correlation matrix
        correlation = pd.DataFrame([
            [1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
            [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
            [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
            [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
            [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
            [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
            [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]
        ], index=countries, columns=countries)

        # Table 2 of the He-Litterman paper: Volatilities
        volatilities = pd.DataFrame([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187], index=countries, columns=["vol"])
        covariance = volatilities.dot(volatilities.T) * correlation

        # Table 2 of the He-Litterman paper: Market-capitalised weights
        market_weights = pd.DataFrame([0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615], index=countries, columns=["CapWeight"])

        # View-2
        bl_model = VanillaBlackLitterman()
        views = [0.05, 0.03]
        pick_list = [
            {
                "DE": 1.0,
                "FR": -market_weights.loc["FR"]/(market_weights.loc["FR"] + market_weights.loc["UK"]),
                "UK": -market_weights.loc["UK"] / (market_weights.loc["FR"] + market_weights.loc["UK"])
            },
            {
                "CA": 1,
                "US": -1
            }
        ]
        bl_model.allocate(covariance=covariance,
                          market_capitalised_weights=market_weights,
                          investor_views=views,
                          pick_list=pick_list,
                          asset_names=covariance.columns,
                          tau=0.05,
                          risk_aversion=2.5)
        expected_returns = (bl_model.posterior_expected_returns * 100).round(1)
        assert list(expected_returns.values[0]) == [4.4, 8.7, 9.5, 11.2, 4.6, 7.0, 7.5]
        weights = calculate_weights(2.5, bl_model.posterior_covariance, bl_model.posterior_expected_returns)
        weights = (weights * 100).round(1).T
        assert list(weights[0]) == [1.5, 41.9, -3.4, 33.6, 11.0, -8.2, 18.8]

    @staticmethod
    def test_original_black_litterman_results_for_third_view():
        # pylint: disable=invalid-name, unsubscriptable-object
        """
        Test and replicate the results of original Black Litterman paper for third view - bullish view on Canada vs US
        """

        def calculate_weights(risk_aversion, covariance, expected_returns):
            return (inv(covariance).dot(expected_returns.T)) / risk_aversion

        countries = ['AU', 'CA', 'FR', 'DE', 'JP', 'UK', 'US']

        # Table 1 of the He-Litterman paper: Correlation matrix
        correlation = pd.DataFrame([
            [1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
            [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
            [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
            [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
            [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
            [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
            [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]
        ], index=countries, columns=countries)

        # Table 2 of the He-Litterman paper: Volatilities
        volatilities = pd.DataFrame([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187], index=countries, columns=["vol"])
        covariance = volatilities.dot(volatilities.T) * correlation

        # Table 2 of the He-Litterman paper: Market-capitalised weights
        market_weights = pd.DataFrame([0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615], index=countries, columns=["CapWeight"])

        # View-2
        bl_model = VanillaBlackLitterman()
        views = [0.05, 0.04]
        pick_list = [
            {
                "DE": 1.0,
                "FR": -market_weights.loc["FR"]/(market_weights.loc["FR"] + market_weights.loc["UK"]),
                "UK": -market_weights.loc["UK"] / (market_weights.loc["FR"] + market_weights.loc["UK"])
            },
            {
                "CA": 1,
                "US": -1
            }
        ]
        bl_model.allocate(covariance=covariance,
                          market_capitalised_weights=market_weights,
                          investor_views=views,
                          pick_list=pick_list,
                          asset_names=covariance.columns,
                          tau=0.05,
                          risk_aversion=2.5)
        expected_returns = (bl_model.posterior_expected_returns * 100).round(1)
        assert list(expected_returns.values[0]) == [4.4, 9.1, 9.5, 11.3, 4.6, 7.0, 7.3]
        weights = calculate_weights(2.5, bl_model.posterior_covariance, bl_model.posterior_expected_returns)
        weights = (weights * 100).round(1).T
        assert list(weights[0]) == [1.5, 53.3, -3.3, 33.1, 11.0, -7.8, 7.3]

    @staticmethod
    def test_idzorek_omega_method():
        """
        Test the Idzorek method for calculating omega matrix.
        """

        tickers = ['A', 'B']
        covariance = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) * 10e-4
        market_cap_weights = [0.44, 0.56]
        views = [0.02, 0.04]
        pick_list = [{'A': 1}, {'B': 1}]

        bl_model = VanillaBlackLitterman()
        bl_model.allocate(covariance=covariance.values,
                          market_capitalised_weights=market_cap_weights,
                          investor_views=views,
                          pick_list=pick_list,
                          omega_method='user_confidences',
                          view_confidences=[0.1, 0.5],
                          asset_names=tickers)
        weights = bl_model.weights.values[0]
        assert np.round(weights[0], 2) == 0.39
        assert np.round(weights[1], 2) == 0.61
        assert len(weights) == 2
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_value_error_on_inconsistent_views_and_pick_list(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when passing different lengths of views and pick list.
        """

        with self.assertRaises(ValueError):
            tickers = ['A', 'B']
            covariance = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) * 10e-4
            market_cap_weights = [0.44, 0.56]
            views = [0.02, 0.04]
            pick_list = [{'A': 1}]

            bl_model = VanillaBlackLitterman()
            bl_model.allocate(covariance=covariance,
                              market_capitalised_weights=market_cap_weights,
                              investor_views=views,
                              pick_list=pick_list)

    def test_value_error_on_unknown_omega_method(self):
        """
        Test ValueError when passing an unknown omega method string.
        """

        with self.assertRaises(ValueError):
            tickers = ['A', 'B']
            covariance = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) * 10e-4
            market_cap_weights = [0.44, 0.56]
            views = [0.02, 0.04]
            pick_list = [{'A': 1}, {'B': 1}]

            bl_model = VanillaBlackLitterman()
            bl_model.allocate(covariance=covariance,
                              market_capitalised_weights=market_cap_weights,
                              investor_views=views,
                              omega_method='unknown_string',
                              pick_list=pick_list)

    def test_value_error_for_no_user_confidences(self):
        """
        Test ValueError when using Idzorek omega method and not passing any user confidences.
        """

        with self.assertRaises(ValueError):
            tickers = ['A', 'B']
            covariance = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) * 10e-4
            market_cap_weights = [0.44, 0.56]
            views = [0.02, 0.04]
            pick_list = [{'A': 1}, {'B': 1}]

            bl_model = VanillaBlackLitterman()
            bl_model.allocate(covariance=covariance,
                              market_capitalised_weights=market_cap_weights,
                              investor_views=views,
                              pick_list=pick_list,
                              omega_method='user_confidences',
                              asset_names=tickers)

    def test_value_error_on_negative_view_confidences(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when user specifies a negative view confidence.
        """

        with self.assertRaises(ValueError):
            tickers = ['A', 'B']
            covariance = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) * 10e-4
            market_cap_weights = [0.44, 0.56]
            views = [0.02, 0.04]
            pick_list = [{'A': 1}, {'B': 1}]

            bl_model = VanillaBlackLitterman()
            bl_model.allocate(covariance=covariance,
                              market_capitalised_weights=market_cap_weights,
                              investor_views=views,
                              pick_list=pick_list,
                              omega_method='user_confidences',
                              view_confidences=[-0.1, 0.5],
                              asset_names=tickers)

    def test_value_error_on_inconsistent_views_and_confidences(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when passing different lengths of views and user confidences.
        """

        with self.assertRaises(ValueError):
            tickers = ['A', 'B']
            covariance = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) * 10e-4
            market_cap_weights = [0.44, 0.56]
            views = [0.02, 0.04]
            pick_list = [{'A': 1}, {'B': 1}]

            bl_model = VanillaBlackLitterman()
            bl_model.allocate(covariance=covariance,
                              market_capitalised_weights=market_cap_weights,
                              investor_views=views,
                              pick_list=pick_list,
                              omega_method='user_confidences',
                              view_confidences=[0.1])

    @staticmethod
    def test_black_litterman_on_no_asset_names():
        """
        Test the weights calculated by the Black Litterman algorithm when no asset names are specified
        """

        tickers = ['A', 'B']
        covariance = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) * 10e-4
        market_cap_weights = [0.44, 0.56]
        views = [0.02, 0.04]
        pick_list = [{'0': 1}, {'1': 1}]

        bl_model = VanillaBlackLitterman()
        bl_model.allocate(covariance=covariance.values,
                          market_capitalised_weights=market_cap_weights,
                          investor_views=views,
                          pick_list=pick_list)
        weights = bl_model.weights.values[0]
        assert len(weights) == 2
        assert np.round(weights[0], 2) == 0.14
        assert np.round(weights[1], 2) == 0.86
        np.testing.assert_almost_equal(np.sum(weights), 1)

    @staticmethod
    def test_passing_user_specified_omega():
        """
        Test calculation of weights when passing a custom omega matrix.
        """

        tickers = ['A', 'B']
        covariance = pd.DataFrame([[46.0, 1.06], [1.06, 5.33]], index=tickers, columns=tickers) * 10e-4
        market_cap_weights = [0.44, 0.56]
        views = [0.02, 0.04]
        pick_list = [{'A': 1}, {'B': 1}]
        omega = np.array([[0.0023, 0], [0., 0.0002665]])

        bl_model = VanillaBlackLitterman()
        bl_model.allocate(covariance=covariance,
                          market_capitalised_weights=market_cap_weights,
                          investor_views=views,
                          pick_list=pick_list,
                          omega=omega,
                          asset_names=tickers)
        weights = bl_model.weights.values[0]
        assert len(weights) == 2
        assert np.round(weights[0], 2) == 0.14
        assert np.round(weights[1], 2) == 0.86
        np.testing.assert_almost_equal(np.sum(weights), 1)
