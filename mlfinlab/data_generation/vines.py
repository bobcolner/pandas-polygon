# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Implementation of generating financial correlation matrices from
"Generating random correlation matrices based on vines and extended onion method"
by Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
https://www.sciencedirect.com/science/article/pii/S0047259X09000876
and "Generating random correlation matrices based partial correlations" by Harry Joe.
https://www.sciencedirect.com/science/article/pii/S0047259X05000886
"""
import numpy as np


def _correlation_from_partial_dvine(partial_correlations, a_beta, b_beta, row, col):
    """
    Calculates a correlation based on partical correlations using the D-vine method.

    It samples from a beta distribution, adjusts it to the range [-1, 1]. Sets this value
    as the starting partial correlation, and follows the D-vine to calculate the final
    correlation.

    :param partial_correlations: (np.array) Matrix of current partial correlations. It is
        modified during this function's execution.
    :param a_beta: (float) Alpha parameter of the beta distribution to sample from.
    :param b_beta: (float) Beta parameter of the beta distribution to sample from.
    :param row: (int) Starting row of the partial correlation matrix.
    :param col: (int) Starting column of the partial correlation matrix.
    :return: (float) Calculated correlation.
    """
    # Sample from beta distribution. Beta is in the range [0, 1] and adjust
    # ranges to [-1, 1].
    beta_sample = np.random.beta(a_beta, b_beta)
    beta_sample = (beta_sample * 2) - 1
    partial_correlations[row + col, col] = beta_sample
    current_corr = beta_sample

    # Calculate correlation based on partial correlations.
    for j in range(row - 1, 0, -1):
        current_corr *= (
            np.sqrt(
                (1 - partial_correlations[j + col, col] ** 2)
                * (1 - partial_correlations[row + col, j + col] ** 2)
            )
            + partial_correlations[j + col, col] * partial_correlations[row + col, j + col]
        )

    return current_corr


def _correlation_from_partial_cvine(partial_correlations, a_beta, b_beta, row, col):
    """
    Calculates a correlation based on partical correlations using the C-vine method.

    It samples from a beta distribution, adjusts it to the range [-1, 1]. Sets this value
    as the starting partial correlation, and follows the C-vine to calculate the final
    correlation.

    :param partial_correlations: (np.array) Matrix of current partial correlations. It is
        modified during this function's execution.
    :param a_beta: (float) Alpha parameter of the beta distribution to sample from.
    :param b_beta: (float) Beta parameter of the beta distribution to sample from.
    :param row: (int) Starting row of the partial correlation matrix.
    :param col: (int) Starting column of the partial correlation matrix.
    :return: (float) Calculated correlation.
    """
    # Sample from beta distribution. Beta is in the range [0, 1] and adjust
    # ranges to [-1, 1].
    beta_sample = np.random.beta(a_beta, b_beta)
    beta_sample = (beta_sample * 2) - 1
    partial_correlations[row, col] = beta_sample
    current_corr = beta_sample

    # Calculate correlation based on partial correlations.
    for j in range(row - 1, -1, -1):
        current_corr *= (
            np.sqrt((1 - partial_correlations[j, col] ** 2) * (1 - partial_correlations[j, row] ** 2))
            + partial_correlations[j, col] * partial_correlations[j, row]
        )

    return current_corr


def _q_vector_correlations(corr_mat, r_factor, dim):
    """
    Sample from unit vector uniformly on the surface of the k_loc-dimensional hypersphere and
    obtains the q vector of correlations.

    :param corr_mat (np.array) Correlation matrix.
    :param r_factor (np.array) R factor vector based on correlation matrix.
    :param dim: (int) Dimension of the hypersphere to sample from.
    :return: (np.array) Q vector of correlations.
    """
    # Sample from unit vector uniformly on the surface of the dim-dimensional hypersphere.
    theta = np.random.randn(dim)
    theta = theta / np.linalg.norm(theta)
    w_mat = r_factor * theta

    # Obtain q vector of correlations.
    eig_val, eig_vec = np.linalg.eig(corr_mat)
    eig_val = np.diag(eig_val)
    r_mat = np.matmul(np.matmul(eig_vec, np.sqrt(eig_val)), np.transpose(eig_vec))
    q_mat_dist = np.matmul(r_mat, w_mat)

    return q_mat_dist


def sample_from_dvine(dim=10, n_samples=1, beta_dist_fixed=None):
    """
    Generates uniform correlation matrices using the D-vine method.

    It is reproduced with modifications from the following paper:
    `Joe, H., 2006. Generating random correlation matrices based on partial correlations.
    Journal of Multivariate Analysis, 97(10), pp.2177-2189.
    <https://www.sciencedirect.com/science/article/pii/S0047259X05000886>`_

    It uses the partial correlation D-vine to generate partial correlations. The partial
    correlations
    are sampled from a uniform beta distribution and adjusted to thr range [-1, 1]. Then these
    partial correlations are converted into raw correlations by using a recursive formula based
    on its location on the vine.

    :param dim: (int) Dimension of correlation matrix to generate.
    :param n_samples: (int) Number of samples to generate.
    :param beta_dist_fixed: (tuple) Overrides the beta distribution parameters. The input is
        two float parameters (alpha, beta), used in the distribution. (None by default)
    :return: (np.array) Generated correlation matrices of shape (n_samples, dim, dim).
    """
    correlation_matrices = []
    for _ in range(n_samples):
        # Initialize parameters of beta distribution.
        if beta_dist_fixed:
            a_beta, b_beta = beta_dist_fixed
        else:
            b_beta = dim / 2
            a_beta = b_beta

        # Initialize correlation matrix.
        partial_correlations = np.zeros((dim, dim))
        corr_mat = np.eye(dim)

        # For each row and column, calculate the correlations based on its partial correlations.
        for k in range(1, dim):
            # Update beta distribution parameter if not overridden.
            if not beta_dist_fixed:
                a_beta -= 1 / 2
                b_beta -= 1 / 2

            # Perform correlation operations on correct row, column locations.
            for i in range(dim - k):
                current_corr = _correlation_from_partial_dvine(
                    partial_correlations, a_beta, b_beta, k, i
                )

                # Store the result in the final matrix.
                corr_mat[k + i, i] = current_corr
                corr_mat[i, k + i] = current_corr

        # If the original beta distribution was overridden, we need to make sure
        # the correlation matrix distribution is permutation-invariant.
        if beta_dist_fixed:
            perm = np.random.permutation(dim)
            corr_mat = np.take(np.take(corr_mat, perm, 0), perm, 1)

        correlation_matrices.append(corr_mat)

    return np.array(correlation_matrices)


def sample_from_cvine(dim=10, eta=2, n_samples=1, beta_dist_fixed=None):
    """
    Generates uniform correlation matrices using the C-vine method.

    It is reproduced with modifications from the following paper:
    `Lewandowski, D., Kurowicka, D. and Joe, H., 2009. Generating random correlation matrices based
    on vines and extended onion method. Journal of multivariate analysis, 100(9), pp.1989-2001.
    <https://www.sciencedirect.com/science/article/pii/S0047259X09000876>`_

    It uses the partial correlation C-vine to generate partial correlations. The partial
    correlations
    are sampled from a uniform beta distribution proportional to its determinant and the factor
    eta.
    and adjusted to thr range [-1, 1]. Then these partial correlations are converted into raw
    correlations by using a recursive formula based on its location on the vine.

    :param dim: (int) Dimension of correlation matrix to generate.
    :param eta: (int) Corresponds to uniform distribution of beta.
        Correlation matrix `S` has a distribution proportional to [det C]^(eta - 1)
    :param n_samples: (int) Number of samples to generate.
    :param beta_dist_fixed: (tuple) Overrides the beta distribution parameters. The input is
        two float parameters (alpha, beta), used in the distribution. (None by default)
    :return: (np.array) Generated correlation matrices of shape (n_samples, dim, dim).
    """
    correlation_matrices = []
    for _ in range(n_samples):
        # Initialize parameters of beta distribution.
        if beta_dist_fixed:
            a_beta, b_beta = beta_dist_fixed
        else:
            b_beta = eta + (dim - 1) / 2
            a_beta = b_beta

        # Initialize correlation matrix.
        partial_correlations = np.zeros((dim, dim))
        corr_mat = np.eye(dim)

        # For each row and column, calculate the correlations based on its partial correlations.
        for k in range(dim - 1):
            # Update beta distribution parameter if not overridden.
            if not beta_dist_fixed:
                a_beta -= 1 / 2
                b_beta -= 1 / 2

            # Perform correlation operations on correct row, column locations.
            for i in range(k + 1, dim):
                current_corr = _correlation_from_partial_cvine(
                    partial_correlations, a_beta, b_beta, k, i
                )

                # Store the result in the final matrix.
                corr_mat[k, i] = current_corr
                corr_mat[i, k] = current_corr

        # If the original beta distribution was overridden, we need to make sure
        # the correlation matrix distribution is permutation-invariant.
        if beta_dist_fixed:
            perm = np.random.permutation(dim)
            corr_mat = np.take(np.take(corr_mat, perm, 0), perm, 1)

        correlation_matrices.append(corr_mat)

    return np.array(correlation_matrices)


def sample_from_ext_onion(dim=10, eta=2, n_samples=1):
    """
    Generates uniform correlation matrices using extended onion method.

    It is reproduced with modifications from the following paper:
    `Lewandowski, D., Kurowicka, D. and Joe, H., 2009. Generating random correlation matrices based
    on vines and extended onion method. Journal of multivariate analysis, 100(9), pp.1989-2001.
    <https://www.sciencedirect.com/science/article/pii/S0047259X09000876>`_

    It uses the extended onion to generate correlations sampled from a uniform beta distribution.
    It starts with a one-dimensional matrix, and it iteratively grows the matrix by adding extra
    rows and columns by sampling from the convex, closed, compact and full-dimensional set on the
    surface of a k-dimensional hypersphere.

    :param dim: (int) Dimension of correlation matrix to generate.
    :param eta: (int) Corresponds to uniform distribution of beta.
        Correlation matrix `S` has a distribution proportional to [det C]^(eta - 1)
    :param n_samples: (int) Number of samples to generate.
    :return: (np.array) Generated correlation matrices of shape (n_samples, dim, dim).
    """
    correlation_matrices = []
    for _ in range(n_samples):
        # Initialize parameters of beta distribution.
        b_beta = eta + (dim - 2) / 2

        # Initialize correlation matrix by sampling from beta distribution
        # and adjust it to range [-1, 1].
        beta_sample = np.random.beta(b_beta, b_beta)
        r_factor = (2 * beta_sample) - 1
        corr_mat = np.array([[1, r_factor], [r_factor, 1]])

        # Grow correlation matrix.
        for k in range(2, dim):
            # Adjust beta parameter and sample from distribution.
            a_beta = (k - 1) / 2
            b_beta -= 1 / 2
            y_beta_sample = np.random.beta(a_beta, b_beta)
            r_factor = np.array([np.sqrt(y_beta_sample)])

            # Sample from unit vector uniformly on the surface of the k-dimensional hypersphere to
            # oObtain q vector of correlations.
            q_mat_dist = _q_vector_correlations(corr_mat, r_factor, k)

            # Extend resulting correlation matrix.
            next_corr = np.zeros((k + 1, k + 1))
            next_corr[:k, :k] = corr_mat
            next_corr[k, k] = 1
            next_corr[k, :k] = q_mat_dist
            next_corr[:k, k] = q_mat_dist

            corr_mat = next_corr

        correlation_matrices.append(corr_mat)

    return np.array(correlation_matrices)
