import numpy as np
import pandas as pd
from mlfinlab_risk_estimators import RiskEstimators, ReturnsEstimators


# Import price data
stock_returns = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)

# Class that have needed functions
risk_estimators = RiskEstimators()
returns_estimators = ReturnsEstimators()

# Finding the MCD estimator on price data
min_cov_det = risk_estimators.minimum_covariance_determinant(stock_prices, price_data=True)

# Finding the Empirical Covariance on price data
empirical_cov = risk_estimators.empirical_covariance(stock_prices, price_data=True)

# Finding the Shrinked Covariances on price data with every method
shrinked_cov = risk_estimators.shrinked_covariance(stock_prices, price_data=True,
                                                   shrinkage_type='all', basic_shrinkage=0.1)

# Finding the Semi-Covariance on price data
semi_cov = risk_estimators.semi_covariance(stock_prices, price_data=True, threshold_return=0)

# Finding the Exponential Covariance on price data and span of 60
exponential_cov = risk_estimators.exponential_covariance(stock_prices, price_data=True,
                                                         window_span=60)

# Relation of number of observations T to the number of variables N (T/N)
tn_relation = stock_prices.shape[0] / stock_prices.shape[1]

# The bandwidth of the KDE kernel
kde_bwidth = 0.01

# Series of returns from series of prices
stock_returns = ret_est.calculate_returns(stock_prices)

# Finding the simple covariance matrix from a series of returns
cov_matrix = stock_returns.cov()

# Finding the Constant Residual Eigenvalue De-noised Сovariance matrix
const_resid_denoised = risk_estimators.denoise_covariance(cov_matrix, tn_relation,
                                                          denoise_method='const_resid_eigen',
                                                          detone=False, kde_bwidth=kde_bwidth)

# Finding the Spectral Clustering De-noised Сovariance matrix
const_resid_denoised = risk_estimators.denoise_covariance(cov_matrix, tn_relation,
                                                          denoise_method='spectral')

# Finding the Targeted Shrinkage De-noised Сovariance matrix
targ_shrink_denoised = risk_estimators.denoise_covariance(cov_matrix, tn_relation,
                                                          denoise_method='target_shrink',
                                                          detone=False, kde_bwidth=kde_bwidth)

# Finding the Constant Residual Eigenvalue De-noised and De-toned Сovariance matrix
const_resid_detoned = risk_estimators.denoise_covariance(cov_matrix, tn_relation,
                                                         denoise_method='const_resid_eigen',
                                                         detone=True, market_component=1,
                                                         kde_bwidth=kde_bwidth)

# Finding the Hierarchical Clustering Filtered Correlation matrix
hierarchical_filtered = risk_estimators.filter_corr_hierarchical(cov_matrix, method='complete',
                                                                 draw_plot=False)