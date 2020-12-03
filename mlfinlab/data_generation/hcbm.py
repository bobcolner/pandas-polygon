# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Implementation of the Hierarchical Correlation Block Model (HCBM) matrix.
"Clustering financial time series: How long is enough?" by Marti, G., Andler, S., Nielsen, F. and Donnat, P.
https://www.ijcai.org/Proceedings/16/Papers/367.pdf
"""
import numpy as np
import pandas as pd
from statsmodels.sandbox.distributions.multivariate import multivariate_t_rvs


def _hcbm_mat_helper(mat, n_low=0, n_high=214, rho_low=0.1, rho_high=0.9, blocks=4, depth=4):
    """
    Helper function for `generate_hcmb_mat` that recursively places rho values to HCBM matrix
    given as an input.

    By using a uniform distribution we select the start and end locations of the blocks in the
    matrix. For each block, we recurse depth times and repeat splitting up the sub-matrix into
    blocks. Each depth level has a unique correlation (rho) values generated from a uniform
    distributions, and bounded by `rho_low` and `rho_high`. This function works as a
    side-effect to the `mat` parameter.

    It is reproduced with modifications from the following paper:
    `Marti, G., Andler, S., Nielsen, F. and Donnat, P., 2016.
    Clustering financial time series: How long is enough?. arXiv preprint arXiv:1603.04017.
    <https://www.ijcai.org/Proceedings/16/Papers/367.pdf>`_

    :param mat: (np.array) Parent HCBM matrix.
    :param n_low: (int) Start location of HCMB matrix to work on.
    :param n_high: (int) End location of HCMB matrix to work on.
    :param rho_low: (float) Lower correlation bound of the matrix. Must be greater or equal
    to 0.
    :param rho_high: (float) Upper correlation bound of the matrix. Must be less or equal to 1.
    :param blocks: (int) Maximum number of blocks to generate per level of depth.
    :param depth: (int) Depth of recursion for generating new blocks.
    """
    # If block is too small or we have reached our max depth level, return.
    if n_high - n_low <= 1 or depth == 0:
        return
    blocks_num = int(np.random.uniform(2, blocks + 1, size=1))
    # Sample from a uniform distribution to get the required block partitions. Adjust to the
    # required boundaries, and make sure the boundaries are enforced.
    partitions = np.random.uniform(0, 1, size=blocks_num)
    partitions = np.ceil(np.cumsum(partitions / np.sum(partitions)) * (n_high - n_low))
    partitions = np.insert(partitions, 0, 0).astype(int) + n_low
    partitions[-1] = n_high

    # Generate the correlation value for all blocks.
    rho_n = np.random.uniform(
        rho_low + np.finfo(np.float).eps,
        (rho_high - rho_low) / depth + rho_low,
        size=len(partitions),
    )
    # Set the block's rho value, and recurse one level down.
    for i in range(1, len(partitions)):
        mat[partitions[i - 1]:partitions[i], partitions[i - 1]:partitions[i]] = rho_n[i - 1]

        _hcbm_mat_helper(mat, partitions[i - 1], partitions[i], rho_n[i - 1], rho_high, blocks, depth - 1)


def generate_hcmb_mat(t_samples, n_size, rho_low=0.1, rho_high=0.9, blocks=4, depth=4, permute=False):
    """
    Generates a Hierarchical Correlation Block Model (HCBM) matrix  of correlation values.

    By using a uniform distribution we select the start and end locations of the blocks in the
    matrix. For each block, we recurse depth times and repeat splitting up the sub-matrix into
    blocks. Each depth level has a unique correlation (rho) values generated from a uniform
    distributions, and bounded by `rho_low` and `rho_high`.

    It is reproduced with modifications from the following paper:
    `Marti, G., Andler, S., Nielsen, F. and Donnat, P., 2016.
    Clustering financial time series: How long is enough?. arXiv preprint arXiv:1603.04017.
    <https://www.ijcai.org/Proceedings/16/Papers/367.pdf>`_

    :param t_samples: (int) Number of HCBM matrices to generate.
    :param n_size: (int) Size of HCBM matrix.
    :param rho_low: (float) Lower correlation bound of the matrix. Must be greater or equal to 0.
    :param rho_high: (float) Upper correlation bound of the matrix. Must be less or equal to 1.
    :param blocks: (int) Number of blocks to generate per level of depth.
    :param depth: (int) Depth of recursion for generating new blocks.
    :param permute: (bool) Whether to permute the final HCBM matrix.
    :return: (np.array) Generated HCBM matrix of shape (t_samples, n_size, n_size).
    """
    hcbm_matrices = []
    for _ in range(t_samples):
        # Initialize HCBM matrix.
        mat = np.full(fill_value=rho_low, shape=(n_size, n_size))

        # Generate HCBM matrix.
        _hcbm_mat_helper(mat, 0, n_size, rho_low, rho_high, blocks, depth)

        # Set diagonal to 1s for it to be a correlation matrix.
        np.fill_diagonal(mat, 1.0)

        # Permute the matrix if needed.
        if permute:
            perm = np.random.permutation(n_size)
            np.take(mat, perm, 0, out=mat)
            np.take(mat, perm, 1, out=mat)

        hcbm_matrices.append(mat)

    return np.array(hcbm_matrices)


def time_series_from_dist(corr, t_samples=1000, dist="normal", deg_free=3):
    """
    Generates a time series from a given correlation matrix.

    It uses multivariate sampling from distributions to create the time series. It supports
    normal and student-t distributions. This method relies and acts as a wrapper for the
    `np.random.multivariate_normal` and
    `statsmodels.sandbox.distributions.multivariate.multivariate_t_rvs` modules.
    `<https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html>`_
    `<https://www.statsmodels.org/stable/sandbox.html?highlight=sandbox#module-statsmodels.sandbox>`_

    It is reproduced with modifications from the following paper:
    `Marti, G., Andler, S., Nielsen, F. and Donnat, P., 2016.
    Clustering financial time series: How long is enough?. arXiv preprint arXiv:1603.04017.
    <https://www.ijcai.org/Proceedings/16/Papers/367.pdf>`_

    :param corr: (np.array) Correlation matrix.
    :param t_samples: (int) Number of samples in the time series.
    :param dist: (str) Type of distributions to use.
        Can take the values ["normal", "student"].
    :param deg_free: (int) Degrees of freedom. Only used for student-t distribution.
    :return: (pd.DataFrame) The resulting time series of shape (len(corr), t_samples).
    """
    # Initialize means.
    means = np.zeros(len(corr))

    # Generate time series based on distribution.
    if dist == "normal":
        series = np.random.multivariate_normal(means, corr, t_samples)
    elif dist == "student":
        series = multivariate_t_rvs(means, corr * ((deg_free - 2) / deg_free), df=deg_free, n=t_samples)
    else:
        raise ValueError("{} is not supported".format(dist))

    return pd.DataFrame(series)
