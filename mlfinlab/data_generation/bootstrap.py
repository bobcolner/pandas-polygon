# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Implementation of generating bootstrapped matrices from
"Bootstrap validation of links of a minimum spanning tree" by F. Musciotto,
L. Marotta, S. Miccichè, and R. N. Mantegna https://arxiv.org/pdf/1802.03395.pdf.
"""

import numpy as np
import pandas as pd


def row_bootstrap(mat, n_samples=1, size=None):
    """
    Uses the Row Bootstrap method to generate a new matrix of size equal or smaller than the given matrix.

    It samples with replacement a random row from the given matrix. If the required bootstrapped
    columns' size is less than the columns of the original matrix, it randomly samples contiguous
    columns of the required size. It cannot generate a matrix greater than the original.

    It is inspired by the following paper:
    `Musciotto, F., Marotta, L., Miccichè, S. and Mantegna, R.N., 2018. Bootstrap validation of
    links of a minimum spanning tree. Physica A: Statistical Mechanics and its Applications,
    512, pp.1032-1043. <https://arxiv.org/pdf/1802.03395.pdf>`_.

    :param mat: (pd.DataFrame/np.array) Matrix to sample from.
    :param n_samples: (int) Number of matrices to generate.
    :param size: (tuple) Size of the bootstrapped matrix.
    :return: (np.array) The generated bootstrapped matrices. Has shape (n_samples, size[0], size[1]).
    """
    if isinstance(mat, pd.DataFrame):
        mat = mat.values

    # If size is not given, use the size of the given matrix.
    if not size:
        size = mat.shape

    gen_mats = []
    for _ in range(n_samples):
        # Sample with replacement the number of rows indices given by the size parameter. Convert
        # it to a matrix of rows indices.
        rows = np.repeat(np.random.choice(size[0], size=size[0]), size[1]).reshape(size)

        cols = []
        # Randomly sample the column indices required based on the size parameter.
        for _ in range(size[0]):
            col_starts = np.random.choice(mat.shape[1] - size[1] + 1)
            cols.append(np.arange(col_starts, col_starts + size[1]))

        # Append the resulting rows and columns.
        gen_mats.append(mat[rows, cols])

    return np.array(gen_mats)


def pair_bootstrap(mat, n_samples=1, size=None):
    """
    Uses the Pair Bootstrap method to generate a new correlation matrix of returns.

    It generates a correlation matrix based on the number of columns of the returns matrix given. It
    samples with replacement a pair of columns from the original matrix, the rows of the pairs generate
    a new row-bootstrapped matrix. The correlation value of the pair of assets is calculated and
    its value is used to fill the corresponding value in the generated correlation matrix.

    It is inspired by the following paper:
    `Musciotto, F., Marotta, L., Miccichè, S. and Mantegna, R.N., 2018. Bootstrap validation of
    links of a minimum spanning tree. Physica A: Statistical Mechanics and its Applications,
    512, pp.1032-1043. <https://arxiv.org/pdf/1802.03395.pdf>`_.

    :param mat: (pd.DataFrame/np.array) Returns matrix to sample from.
    :param n_samples: (int) Number of matrices to generate.
    :param size: (int) Size of the bootstrapped correlation matrix.
    :return: (np.array) The generated bootstrapped correlation matrices. Has shape (n_samples, mat.shape[1], mat.shape[1]).
    """
    if isinstance(mat, pd.DataFrame):
        mat = mat.values

    # If size is not given, use the size of the given matrix.
    if not size:
        size = mat.shape
    n_assets = size[1]
    gen_mats = []
    for _ in range(n_samples):
        boot_mat = np.diag(np.ones(n_assets))

        # Fill the upper triangular of the bootstrapped correlation matrix with the correlation
        # value of the pairs.
        for row in range(n_assets):
            for col in range(row + 1, n_assets):
                # Sample two random assets.
                bootstrap_pairs = mat[:, [row, col]]

                # Generate a new row-bootstrapped matrix.
                bootstrap_pairs = row_bootstrap(bootstrap_pairs, n_samples=1)[0]

                # Calculate the correlation value of the pairs.
                dep_mat = np.corrcoef(np.transpose(bootstrap_pairs))

                # Fill the correlation matrix with the found correlation value.
                boot_mat[row, col] = dep_mat[0, 1]
                boot_mat[col, row] = dep_mat[0, 1]

        # Append the resulting rows and columns.
        gen_mats.append(boot_mat)

    return np.array(gen_mats)


def block_bootstrap(mat, n_samples=1, size=None, block_size=None):
    """
    Uses the Block Bootstrap method to generate a new matrix of size equal to or smaller than the given matrix.

    It divides the original matrix into blocks of the given size. It samples with replacement random
    blocks to populate the bootstrapped matrix. It cannot generate a matrix greater than the original.

    It is inspired by the following paper:
    `Künsch, H.R., 1989. The jackknife and the bootstrap for general stationary observations.
    Annals of Statistics, 17(3), pp.1217-1241. <https://projecteuclid.org/euclid.aos/1176347265>`_.

    :param mat: (pd.DataFrame/np.array) Matrix to sample from.
    :param n_samples: (int) Number of matrices to generate.
    :param size: (tuple) Size of the bootstrapped matrix.
    :param block_size: (tuple) Size of the blocks.
    :return: (np.array) The generated bootstrapped matrices. Has shape (n_samples, size[0], size[1]).
    """
    if isinstance(mat, pd.DataFrame):
        mat = mat.values

    # If size is not given, use the size of the given matrix.
    if not size:
        size = mat.shape

    # If the block size is not given, use a block size of 10% of givem matrix size. Each dimension
    # has a minimum size of 2.
    if not block_size:
        block_size = (max(int(np.ceil(size[0] * 0.1)), 2), max(int(np.ceil(size[1] * 0.1)), 2))

    # Calculate how many blocks need to be sampled to generate the bootstrapped matrix.
    rows_blocks = int(np.ceil(size[0] / block_size[0]))
    cols_blocks = int(np.ceil(size[1] / block_size[1]))

    gen_mats = []
    for _ in range(n_samples):
        # Initialize a matrix capable of holding all blocks needed.
        boot_mat = np.zeros((rows_blocks * block_size[0], cols_blocks * block_size[1]))

        # For each block sampled, add to the corresponding row-columns in the bootstrap matrix.
        for row in range(rows_blocks):
            for col in range(cols_blocks):
                # Get the row and columns location of the block, adjusted by its size.
                row_block_loc = np.random.choice(rows_blocks) * block_size[0]
                col_block_loc = np.random.choice(cols_blocks) * block_size[1]
                bootstrap_block = mat[row_block_loc: row_block_loc + block_size[0], col_block_loc: col_block_loc + block_size[1]]

                # If the resulting block is not the shape of the required block, it means it hit
                # and edge, need to adjust the row/column location to grab the required block size.
                # This results in an overlapping block.
                if bootstrap_block.shape != block_size:
                    row_block_loc -= block_size[0] - bootstrap_block.shape[0]
                    col_block_loc -= block_size[1] - bootstrap_block.shape[1]
                    bootstrap_block = mat[row_block_loc: row_block_loc + block_size[0], col_block_loc: col_block_loc + block_size[1]]

                # Assign the sampled block to the correct location in the bootstrap matrix.
                row_loc = row * block_size[0]
                col_loc = col * block_size[1]
                boot_mat[row_loc : row_loc + block_size[0], col_loc : col_loc + block_size[1]] = bootstrap_block

        # Adjust the final matrix to the required size.
        boot_mat = boot_mat[: size[0], : size[1]]

        gen_mats.append(boot_mat)

    return np.array(gen_mats)
