# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
#pylint: disable=missing-docstring
import pandas as pd
import numpy as np
from scipy.stats.distributions import chi2
from mlfinlab.portfolio_optimization.modern_portfolio_theory import MeanVarianceOptimisation


class RobustBayesianAllocation:
    """
    This class implements the Robust Bayesian Allocation (RBA) algorithm from the following paper: Meucci, A., 2011. Robust Bayesian Allocation.
    Instead of relying on historical sample data, this method combines information from the sample distribution and investor-specified prior distribution
    to calculate the posterior market distribution. Finally, the algorithm generates a Bayesian efficient frontier using this posterior and selects the
    robust bayesian portfolio from it.
    """

    def __init__(self, discretisations=10):
        """
        Initialise.

        Class Variables:

        - ``discretisations`` - (int) Number of portfolios to generate along the bayesian efficient frontier. The final robust portfolio
                                      will be chosen from the set of these portfolios.
        - ``weights`` - (pd.DataFrame) Final portfolio weights.
        - ``portfolio_return`` - (float) Portfolio return.
        - ``portfolio_risk`` - (float) Portfolio variance/risk.
        - ``posterior_mean`` - (pd.DataFrame) Posterior mean returns for assets in portfolio.
        - ``posterior_covariance`` - (pd.DataFrame) Posterior covariance matrix of asset returns.
        - ``posterior_mean_confidence`` - (float) Investor confidence in the posterior mean distribution.
        - ``posterior_covariance_confidence`` - (float) Investor confidence in the posterior covariance distribution.
        """

        self.discretisations = discretisations
        self.weights = None
        self.portfolio_return = None
        self.portfolio_risk = None
        self.posterior_mean = None
        self.posterior_covariance = None
        self.posterior_mean_confidence = None
        self.posterior_covariance_confidence = None

    def allocate(self, prior_mean, prior_covariance, sample_mean, sample_covariance, sample_size, relative_confidence_in_prior_mean=1,
                 relative_confidence_in_prior_covariance=1, posterior_mean_estimation_risk_level=0.1, posterior_covariance_estimation_risk_level=0.1,
                 max_volatility=1.0, asset_names=None):
        #pylint: disable=too-many-arguments, invalid-name
        """
        Combines the prior and sample distributions to calculate a robust bayesian portfolio.

        :param prior_mean: (Numpy array/Python list) The mean returns of the prior distribution.
        :param prior_covariance: (pd.DataFrame/Numpy matrix) The covariance of the prior distribution.
        :param sample_mean: (Numpy array/Python list) The mean returns of sample distribution.
        :param sample_covariance: (pd.DataFrame/Numpy matrix) The covariance of returns of the sample distribution.
        :param sample_size: (int) Number of observations in the data used to estimate sample means and covariance.
        :param relative_confidence_in_prior_mean: (float) A numeric value specifying the investor's confidence in the mean of the prior distribution.
                                                          This confidence is measured relative to the sample distribution.
        :param relative_confidence_in_prior_covariance: (float) A numeric value specifying the investor's confidence in the covariance of the prior
                                                                distribution. This confidence is measured relative to the sample distribution.
        :param posterior_mean_estimation_risk_level: (float) Denotes the confidence of investor to estimation risk in posterior mean. Lower value corresponds
                                                             to less confidence and a more aggressive investor while a higher value will result in a more
                                                             conservative portfolio.
        :param posterior_covariance_estimation_risk_level: (float) Denotes the confidence of investor to estimation risk in posterior covariance. Lower value
                                                                   corresponds to less confidence and a more aggressive investor while a higher value will
                                                                   result in a more conservative portfolio.
        :param max_volatility: (float) The maximum preferred volatility of the final robust portfolio.
        :param asset_names: (Numpy array/Python list) List of asset names in the portfolio.
        """

        self._error_checks(prior_mean, prior_covariance, sample_mean, sample_covariance, sample_size,
                           relative_confidence_in_prior_mean, relative_confidence_in_prior_covariance,
                           posterior_mean_estimation_risk_level, posterior_covariance_estimation_risk_level,
                           max_volatility)

        num_assets = len(prior_mean)
        if asset_names is None:
            if isinstance(sample_covariance, pd.DataFrame):
                asset_names = sample_covariance.columns
            elif isinstance(prior_covariance, pd.DataFrame):
                asset_names = prior_covariance.columns
            else:
                asset_names = list(map(str, range(num_assets)))
        prior_mean, prior_covariance, sample_mean, sample_covariance = \
            self._pre_process_inputs(prior_mean, prior_covariance, sample_mean, sample_covariance)

        # Calculate the posterior market distributions
        self._calculate_posterior_distribution(sample_size, relative_confidence_in_prior_mean, relative_confidence_in_prior_covariance,
                                               sample_mean, sample_covariance, prior_mean, prior_covariance)

        # Get portfolios along Bayesian Efficient Frontier
        bayesian_portfolios, bayesian_portfolio_volatilities, bayesian_portfolio_returns = self._calculate_bayesian_frontier(asset_names)

        # Calculate gamma values
        gamma_mean, gamma_covariance = self._calculate_gamma(posterior_mean_estimation_risk_level, posterior_covariance_estimation_risk_level,
                                                             max_volatility, num_assets)

        # Find the robust Bayesian portfolio.
        self._find_robust_portfolio(bayesian_portfolios, bayesian_portfolio_volatilities,
                                    bayesian_portfolio_returns, gamma_mean, gamma_covariance, asset_names)

    @staticmethod
    def _pre_process_inputs(prior_mean, prior_covariance, sample_mean, sample_covariance):
        """
        Initial preprocessing of inputs.

        :param prior_mean: (Numpy array/Python list) The mean returns of the prior distribution.
        :param prior_covariance: (pd.DataFrame/Numpy matrix) The covariance of the prior distribution.
        :param sample_mean: (Numpy array/Python list) The mean returns of sample distribution.
        :param sample_covariance: (pd.DataFrame/Numpy matrix) The covariance of returns of the sample distribution.
        :return: (Numpy array, Numpy matrix, Numpy array, Numpy matrix) Same inputs but converted to numpy arrays and matrices.
        """

        prior_mean = np.reshape(prior_mean, (len(prior_mean), 1))
        sample_mean = np.reshape(sample_mean, (len(sample_mean), 1))

        if isinstance(prior_covariance, pd.DataFrame):
            prior_covariance = prior_covariance.values
        if isinstance(sample_covariance, pd.DataFrame):
            sample_covariance = sample_covariance.values

        return prior_mean, prior_covariance, sample_mean, sample_covariance

    def _find_robust_portfolio(self, bayesian_portfolios, bayesian_portfolio_volatilities, bayesian_portfolio_returns, gamma_mean,
                               gamma_covariance, asset_names):
        """
        From the set of portfolios along the bayesian efficient frontier, select the robust portfolio - one which gives highest return for highest risk.

        :param bayesian_portfolios: (Python list) List of portfolio weights along the bayesian efficient frontier
        :param bayesian_portfolio_volatilities: (Python list) Volatilities of portfolios along the bayesian efficient frontier
        :param bayesian_portfolio_returns: (Python list) Expected returns of portfolios along the bayesian efficient frontier
        :param gamma_mean: (float) Gamma value for the mean.
        :param gamma_covariance: (float) Gamma value for the covariance.
        :param asset_names: (Numpy array/Python list) List of asset names.
        """

        target_weights = None
        target_volatility = None
        target_return = float("-inf")
        for index in range(self.discretisations):

            # Risk constraint
            if bayesian_portfolio_volatilities[index] <= gamma_covariance:

                # Calculate the portfolio objective
                objective = bayesian_portfolio_returns[index] - gamma_mean * np.sqrt(
                    bayesian_portfolio_volatilities[index])

                # Select portfolio with maximum return
                if objective > target_return:
                    target_weights = bayesian_portfolios[index]
                    target_return = bayesian_portfolio_returns[index]
                    target_volatility = bayesian_portfolio_volatilities[index]

        if target_weights is None:
            raise ValueError("No robust portfolio found within credibility set. Try increasing max_volatility or adjusting risk level parameters.")

        self.weights = target_weights
        self.weights = pd.DataFrame(self.weights, columns=asset_names)
        self.portfolio_return = target_return
        self.portfolio_risk = target_volatility

    def _calculate_gamma(self, posterior_mean_estimation_risk_level, posterior_covariance_estimation_risk_level, max_volatility, num_assets):
        # pylint: disable=invalid-name
        """
        Calculate the gamma values appearing in the robust bayesian allocation objective and risk constraint.

        :param posterior_mean_estimation_risk_level: (float) Denotes the confidence of investor to estimation risk in posterior mean. Lower value corresponds
                                                             to less confidence and a more conservative investor while a higher value will result in a more
                                                             risky portfolio.
        :param posterior_covariance_estimation_risk_level: (float) Denotes the confidence of investor to estimation risk in posterior covariance. Lower value
                                                                   corresponds to less confidence and a more conservative investor while a higher value will
                                                                   result in a more risky portfolio.
        :param max_volatility: (float) The maximum preferred volatility of the final robust portfolio.
        :param num_assets: (int) Number of assets in the portfolio.
        :return: (float, float) gamma mean, gamma covariance
        """

        mean_risk_aversion = chi2.ppf(posterior_mean_estimation_risk_level, num_assets)
        gamma_mean = np.sqrt((mean_risk_aversion / self.posterior_mean_confidence) *
                             (self.posterior_covariance_confidence / (self.posterior_covariance_confidence - 2)))

        covariance_risk_aversion = chi2.ppf(posterior_covariance_estimation_risk_level,
                                            num_assets * (num_assets + 1) / 2)
        gamma_covariance = max_volatility / \
                           (self.posterior_covariance_confidence / (self.posterior_covariance_confidence + num_assets + 1) +
                            np.sqrt(2 * self.posterior_covariance_confidence * self.posterior_covariance_confidence * covariance_risk_aversion /
                                    ((self.posterior_covariance_confidence + num_assets + 1) ** 3)))
        return gamma_mean, gamma_covariance

    def _calculate_posterior_distribution(self, sample_size, relative_confidence_in_prior_mean, relative_confidence_in_prior_covariance, sample_mean,
                                          sample_covariance, prior_mean, prior_covariance):
        """
        Calculate the posterior market distribution from prior and sample distributions.

        :param sample_size: (int) Number of observations in the data used to estimate sample means and covariance.
        :param relative_confidence_in_prior_mean: (float) A numeric value specifying the investor's confidence in the mean of the prior distribution.
                                                          This confidence is measured relative to the sample distribution.
        :param relative_confidence_in_prior_covariance: (float) A numeric value specifying the investor's confidence in the covariance of the prior
                                                                distribution. This confidence is measured relative to the sample distribution.
        :param sample_mean: (Numpy array) The mean returns of sample distribution.
        :param sample_covariance: (Numpy matrix) The covariance of returns of the sample distribution.
        :param prior_mean: (Numpy array) The mean returns of the prior distribution.
        :param prior_covariance: (Numpy matrix) The covariance of the prior distribution.
        """

        sample_confidence = sample_size
        prior_mean_confidence = sample_size * relative_confidence_in_prior_mean
        prior_covariance_confidence = sample_size * relative_confidence_in_prior_covariance

        self.posterior_mean_confidence = sample_confidence + prior_mean_confidence
        self.posterior_covariance_confidence = sample_confidence + prior_covariance_confidence

        self.posterior_mean = (1 / self.posterior_mean_confidence) * (prior_mean_confidence * prior_mean + sample_confidence * sample_mean)
        self.posterior_covariance = (1 / self.posterior_covariance_confidence) * \
                                    (prior_covariance_confidence * prior_covariance + sample_confidence * sample_covariance +
                                     ((prior_mean - sample_mean).dot((prior_mean - sample_mean).T) / (1 / sample_confidence + 1 / prior_mean_confidence)))

    def _calculate_bayesian_frontier(self, asset_names):
        """
        Generate portfolios along the bayesian efficient frontier.

        :param asset_names: (Numpy array/Python list) List of asset names in the portfolio.
        :return: (Python list, Python list, Python list) Portfolios along the bayesian efficient frontier.
        """

        # Calculate minimum risk portfolio
        mvo = MeanVarianceOptimisation()
        mvo.allocate(expected_asset_returns=self.posterior_mean,
                     covariance_matrix=self.posterior_covariance,
                     asset_names=asset_names,
                     solution='min_volatility')
        min_volatility_weights = mvo.weights.values
        min_volatility_return = mvo.portfolio_return

        # Maximum return
        maximum_return = np.max(self.posterior_mean)

        # Get the target returns along the frontier
        step_size = (maximum_return - min_volatility_return) / (self.discretisations - 1)
        target_returns = np.arange(min_volatility_return, maximum_return + step_size, step_size)

        # Start calculating the portfolios along the frontier
        bayesian_portfolios = [min_volatility_weights]
        bayesian_portfolio_volatilities = [mvo.portfolio_risk]
        bayesian_portfolio_returns = [mvo.portfolio_return]
        for target_return in target_returns:
            mvo.allocate(expected_asset_returns=self.posterior_mean,
                         covariance_matrix=self.posterior_covariance,
                         target_return=target_return,
                         asset_names=asset_names,
                         solution='efficient_risk')
            bayesian_portfolios.append(mvo.weights.values)
            bayesian_portfolio_volatilities.append(mvo.portfolio_risk)
            bayesian_portfolio_returns.append(mvo.portfolio_return)

        return bayesian_portfolios, bayesian_portfolio_volatilities, bayesian_portfolio_returns

    @staticmethod
    def _error_checks(prior_mean, prior_covariance, sample_mean, sample_covariance, sample_size, relative_confidence_in_prior_mean,
                      relative_confidence_in_prior_covariance, posterior_mean_estimation_risk_level, posterior_covariance_estimation_risk_level,
                      max_volatility):
        #pylint: disable=invalid-name
        """
        Initial error checks on inputs.

        :param prior_mean: (Numpy array/Python list) The mean returns of the prior distribution.
        :param prior_covariance: (pd.DataFrame/Numpy matrix) The covariance of the prior distribution.
        :param sample_mean: (Numpy array/Python list) The mean returns of sample distribution.
        :param sample_covariance: (pd.DataFrame/Numpy matrix) The covariance of returns of the sample distribution.
        :param sample_size: (int) Number of observations in the data used to estimate sample means and covariance.
        :param relative_confidence_in_prior_mean: (float) A numeric value specifying the investor's confidence in the mean of the prior distribution.
                                                          This confidence is measured relative to the sample distribution.
        :param relative_confidence_in_prior_covariance: (float) A numeric value specifying the investor's confidence in the covariance of the prior
                                                                distribution. This confidence is measured relative to the sample distribution.
        :param posterior_mean_estimation_risk_level: (float) Denotes the confidence of investor to estimation risk in posterior mean. Lower value corresponds
                                                             to less confidence and a more conservative investor while a higher value will result in a more
                                                             risky portfolio.
        :param posterior_covariance_estimation_risk_level: (float) Denotes the aversion of investor to estimation risk in posterior covariance. Lower value
                                                                   corresponds to less confidence and a more conservative investor while a higher value will
                                                                   result in a more risky portfolio.
        :param max_volatility: (float) The maximum preferred volatility of the final robust portfolio.
        """

        if len(prior_mean) != prior_covariance.shape[0]:
            raise ValueError("Length of prior mean and prior covariance does not match.")

        if len(sample_mean) != sample_covariance.shape[0]:
            raise ValueError("Length of sample mean and sample covariance does not match.")

        if len(prior_mean) != len(sample_mean):
            raise ValueError("Length of prior mean and sample mean does not match.")

        if relative_confidence_in_prior_covariance < 0 or relative_confidence_in_prior_mean < 0:
            raise ValueError("Confidence in prior cannot be negative. Please specify a value larger than 0.")

        if posterior_mean_estimation_risk_level < 0 or posterior_covariance_estimation_risk_level < 0:
            raise ValueError("Posterior estimation risk confidence cannot be negative. Please specify a value larger than 0.")

        if max_volatility < 0:
            raise ValueError("Maximum volatility cannot be negative. Please specify a value larger than 0.")

        if sample_size < 2:
            raise ValueError("The optimisation does not work with only 1 asset. Please specify at least 2 assets in the portfolio.")
