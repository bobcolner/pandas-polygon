# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Contains methods for generating correlated random walks.
"""

import numpy as np
import pandas as pd


def generate_cluster_time_series(n_series, t_samples=100, k_corr_clusters=1,
                                 d_dist_clusters=1, rho_main=0.1, rho_corr=0.3, price_start=100.0,
                                 dists_clusters=("normal", "normal", "student-t", "normal", "student-t")):
    """
    Generates a synthetic time series of correlation and distribution clusters.

    It is reproduced with modifications from the following paper:
    `Donnat, P., Marti, G. and Very, P., 2016. Toward a generic representation of random
    variables for machine learning. Pattern Recognition Letters, 70, pp.24-31.
    <https://www.sciencedirect.com/science/article/pii/S0167865515003906>`_

    `www.datagrapple.com. (n.d.). DataGrapple - Tech: A GNPR tutorial: How to cluster random walks.
    [online] Available at:  [Accessed 26 Aug. 2020].
    <https://www.datagrapple.com/Tech/GNPR-tutorial-How-to-cluster-random-walks.html>`_

    This method creates `n_series` time series of length `t_samples`. Each time series is divided
    into `k_corr_clusters` correlation clusters. Each correlation cluster is subdivided into
    `d_dist_clusters` distribution clusters.
    A main distribution is sampled from a normal distribution with mean = 0 and stdev = 1, adjusted
    by a `rho_main` factor. The correlation clusters are sampled from a given distribution, are generated
    once, and adjusted by a `rho_corr` factor. The distribution clusters are sampled from other
    given distributions, and adjusted by (1 - `rho_main` - `rho_corr`). They are sampled for each time series.
    These three series are added together to form a time series of returns. The final time series
    is the cumulative sum of the returns, with a start price given by `price_start`.

    :param n_series: (int) Number of time series to generate.
    :param t_samples: (int) Number of samples in each time series.
    :param k_corr_clusters: (int) Number of correlation clusters in each time series.
    :param d_dist_clusters: (int) Number of distribution clusters in each time series.
    :param rho_main: (float): Strength of main time series distribution.
    :param rho_corr: (float): Strength of correlation cluster distribution.
    :param price_start: (float) Starting price of the time series.
    :param dists_clusters: (list) List containing the names of the distributions to sample from.
        The following numpy distributions are available: "normal" = normal(0, 1), "normal_2" = normal(0, 2),
        "student-t" = standard_t(3)/sqrt(3), "laplace" = laplace(1/sqrt(2)). The first disitribution
        is used to sample for the correlation clusters (k_corr_clusters), the remaining ones are used
        to sample for the distribution clusters (d_dist_clusters).
    :return: (pd.DataFrame) Generated time series. Has size (t_samples, n_series).
    """
    dists = {
        "normal": np.random.normal,
        "normal_2": np.random.normal,
        "student-t": np.random.standard_t,
        "laplace": np.random.laplace,
    }
    dists_params = {
        "normal": [0, 1],
        "normal_2": [0, 2],
        "student-t": [3],
        "laplace": [0, 1 / np.sqrt(2)],
    }
    # Generate main time series distribution.
    main_dist = np.sqrt(rho_main) * np.random.normal(0, 1, size=t_samples)

    # Generate distribution of correlation clusters.
    correlation_dist = np.sqrt(rho_corr) * dists[dists_clusters[0]](
        *dists_params[dists_clusters[0]], size=(k_corr_clusters, t_samples)
    )
    if dists_clusters[0] == "student-t":
        correlation_dist /= np.sqrt(3)

    # Generate time series.
    rho_dist = np.sqrt(1 - rho_main - rho_corr)
    time_series = []
    num_clusters = int(k_corr_clusters * d_dist_clusters * 2)
    for i in range(n_series):
        # Sample from the required correlation cluster distribution and adjust it by beta.
        corr_cluster_sample = correlation_dist[
            int(np.ceil((i + 1) * k_corr_clusters / n_series)) - 1
        ]

        # Sample from the required distribution cluster distribution.
        d_index = (((i + 1) // num_clusters) % 4) + 1
        dist_cluster_sample = rho_dist * dists[dists_clusters[d_index]](
            *dists_params[dists_clusters[d_index]], size=t_samples
        )
        if dists_clusters[d_index] == "student-t":
            dist_cluster_sample /= np.sqrt(3)

        # Generate resulting time series.
        series = main_dist + corr_cluster_sample + dist_cluster_sample
        series[0] += price_start
        time_series.append(np.cumsum(series))

    return pd.DataFrame(np.transpose(time_series))
