# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Implementation of sampling realistic financial correlation matrices from
"CorrGAN: Sampling Realistic Financial Correlation Matrices using
Generative Adversarial Networks" by Gautier Marti.
https://arxiv.org/pdf/1910.09504.pdf
"""
from os import listdir, path
import numpy as np
from scipy.cluster import hierarchy
from statsmodels.stats.correlation_tools import corr_nearest


def sample_from_corrgan(model_loc, dim=10, n_samples=1):
    # pylint: disable=import-outside-toplevel, disable=too-many-locals
    """
    Samples correlation matrices from the pre-trained CorrGAN network.

    It is reproduced with modifications from the following paper:
    `Marti, G., 2020, May. CorrGAN: Sampling Realistic Financial Correlation Matrices Using
    Generative Adversarial Networks. In ICASSP 2020-2020 IEEE International Conference on
    Acoustics, Speech and Signal Processing (ICASSP) (pp. 8459-8463). IEEE.
    <https://arxiv.org/pdf/1910.09504.pdf>`_

    It loads the appropriate CorrGAN model for the required dimension. Generates a matrix output
    from this network. Symmetries this matrix and finds the nearest correlation matrix
    that is positive semi-definite. Finally, it maximizes the sum of the similarities between
    adjacent leaves to arrange it with hierarchical clustering.

    The CorrGAN network was trained on the correlation profiles of the S&P 500 stocks. Therefore
    the output retains these properties. In addition, the final output retains the following
    6 stylized facts:

    1. Distribution of pairwise correlations is significantly shifted to the positive.

    2. Eigenvalues follow the Marchenko-Pastur distribution, but for a very large first
    eigenvalue (the market).

    3. Eigenvalues follow the Marchenko-Pastur distribution, but for a couple of other
    large eigenvalues (industries).

    4. Perron-Frobenius property (first eigenvector has positive entries).

    5. Hierarchical structure of correlations.

    6. Scale-free property of the corresponding Minimum Spanning Tree (MST).

    :param model_loc: (str) Location of folder containing CorrGAN models.
    :param dim: (int) Dimension of correlation matrix to sample.
        In the range [2, 200].
    :param n_samples: (int) Number of samples to generate.
    :return: (np.array) Sampled correlation matrices of shape (n_samples, dim, dim).
    """
    # Import here needed to prevent unnecessary imports in other parts of code.
    import tensorflow as tf

    # Validate dimension.
    if not (1 < dim <= 200):
        raise ValueError("Dimension not supported, {}".format(dim))

    # Resulting correlation matrices.
    nearest_corr_mats = []

    # Load generator model closest to the required dimension by looking at the models folder.
    dimension_from_folder = [
        int(f.split("_")[1][:-1])
        for f in listdir(model_loc)
        if not path.isfile(path.join(model_loc, f))
    ]
    all_generator_dimensions = np.sort(dimension_from_folder)
    closest_dimension = next(filter(lambda i: i >= dim, all_generator_dimensions))

    # Load model.
    generator = tf.keras.models.load_model(
        "{}/generator_{}d".format(model_loc, closest_dimension), compile=False
    )

    # Sample from generator. Input dimension based on network.
    noise_dim = generator.layers[0].input_shape[1]
    noise = tf.random.normal([n_samples, noise_dim])
    generated_mat = generator(noise, training=False)

    # Get the indices of an upper triangular matrix.
    tri_rows, tri_cols = np.triu_indices(dim, k=1)

    # For each sample generated, make them strict correlation matrices
    # by projecting them on the nearest correlation matrix using Higham’s
    # alternating projections method.
    for i in range(n_samples):
        # Grab only the required dimensions from generated matrix.
        corr_mat = np.array(generated_mat[i, :dim, :dim, 0])

        # Set diagonal to 1 and symmetrize.
        np.fill_diagonal(corr_mat, 1)
        corr_mat[tri_cols, tri_rows] = corr_mat[tri_rows, tri_cols]
        # Get nearest correlation matrix that is positive semi-definite.
        nearest_corr_mat = corr_nearest(corr_mat)

        # Set diagonal to 1 and symmetrize.
        np.fill_diagonal(nearest_corr_mat, 1)
        nearest_corr_mat[tri_cols, tri_rows] = nearest_corr_mat[tri_rows, tri_cols]

        # Arrange with hierarchical clustering by maximizing the sum of the
        # similarities between adjacent leaves.
        dist = 1 - nearest_corr_mat
        linkage_mat = hierarchy.linkage(dist[tri_rows, tri_cols], method="ward")
        optimal_leaves = hierarchy.optimal_leaf_ordering(linkage_mat, dist[tri_rows, tri_cols])
        optimal_ordering = hierarchy.leaves_list(optimal_leaves)
        ordered_corr = nearest_corr_mat[optimal_ordering, :][:, optimal_ordering]
        nearest_corr_mats.append(ordered_corr)

    return np.array(nearest_corr_mats)
