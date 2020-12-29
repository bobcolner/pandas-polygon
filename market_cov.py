import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.estimators.risk_estimators import RiskEstimators
from mlfinlab.codependence import get_distance_matrix


risk_estimators = RiskEstimators()


def get_cov(data: pd.DataFrame, method: str, price_data: bool) -> tuple:
    # price_data: is the data price series or returns series?
    if method == 'empirical':
        cov_mat = risk_estimators.empirical_covariance(data, price_data)
    elif method == 'mcd':
        cov_mat = risk_estimators.minimum_covariance_determinant(data, price_data)
    elif method == 'shrinked_basic':
        cov_mat = risk_estimators.shrinked_covariance(data, price_data, shrinkage_type='basic')
    elif method == 'shrinked_lw':  # Ledoit-Wolf
        cov_mat = risk_estimators.shrinked_covariance(data, price_data, shrinkage_type='lw')
    elif method == 'shrinked_oas':  # Oracle Approximating Shrinkage
        cov_mat = risk_estimators.shrinked_covariance(data, price_data, shrinkage_type='oas')
    elif method == 'semi_covariance':
        cov_mat = risk_estimators.semi_covariance(data, price_data, threshold_return=0)
    elif method == 'exponential':
        cov_mat = risk_estimators.exponential_covariance(data, price_data, window_span=60)
    
    cov_mat = pd.DataFrame(cov_mat).set_index(data.columns)
    cov_mat.columns = data.columns

    cor_mat = risk_estimators.cov_to_corr(cov_mat)
    return cov_mat, cor_mat


def denoise_cov(cov_mat: pd.DataFrame, method: str, rows_per_col: float, detone: bool=True) -> tuple:
    kbw = 0.01
    n_comp = 1
    if method == 'const_resid_eigen':
    # Finding the Constant Residual Eigenvalue De-noised Сovariance matrix
        denoised_cov = risk_estimators.denoise_covariance(cov_mat, rows_per_col,
            denoise_method='const_resid_eigen', detone=detone, market_component=n_comp, kde_bwidth=kbw)
    elif method == 'spectral':
        # Finding the Spectral Clustering De-noised Сovariance matrix
        denoised_cov = risk_estimators.denoise_covariance(cov_mat, rows_per_col,
            denoise_method='spectral', detone=detone, market_component=n_comp)
    elif method == 'const_resid_eigen':
        # Finding the Targeted Shrinkage De-noised Сovariance matrix
        denoised_cov = risk_estimators.denoise_covariance(cov_mat, rows_per_col,
            denoise_method='target_shrink', detone=detone, market_component=n_comp, kde_bwidth=kbw)
    elif method == 'const_resid_eigen':
        # Finding the Constant Residual Eigenvalue De-noised and De-toned Сovariance matrix
        denoised_cov = risk_estimators.denoise_covariance(cov_mat, rows_per_col,
            denoise_method='const_resid_eigen', detone=detone, market_component=n_comp, kde_bwidth=kbw)
    elif method == 'filter_corr_hierarchical':
        # Finding the Hierarchical Clustering Filtered Correlation matrix
        denoised_cov = risk_estimators.filter_corr_hierarchical(cov_mat,
            method='complete', draw_plot=False)

    denoised_cov = pd.DataFrame(denoised_cov).set_index(cov_mat.columns)
    denoised_cov.columns = cov_mat.columns
    denoised_cor = risk_estimators.cov_to_corr(denoised_cov)
    return denoised_cov, denoised_cor


def cov_denoise_detone_dist(data: pd.DataFrame, cov_method: str='shrinked_lw',
    dn_method: str='const_resid_eigen', detone: bool=False) -> tuple:

    cov_mat, cor_mat = get_cov(
        data=data,
        method=cov_method,
        price_data=False
        )
    dncov_mat, dncor_mat = denoise_cov(
        cov_mat=cov_mat,
        method=dn_method,  # const_resid_eigen, target_shrink, spectral
        rows_per_col=data.shape[0] / data.shape[1],
        detone=detone,
        )
    dist_mat = get_distance_matrix(
        X=dncor_mat, 
        distance_metric='absolute_angular'
        )
    return dist_mat, dncor_mat
