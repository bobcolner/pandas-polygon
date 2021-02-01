import numpy as np
import pandas as pd


def random_walk_with_outliers(origin, n_steps, perc_outliers=0.0, outlier_mult=20, seed=42):

    assert (perc_outliers >= 0.0) & (perc_outliers <= 1.0)

    #set seed for reproducibility
    np.random.seed(seed)

    # possible steps
    steps = [-1, 1]

    # simulate steps
    steps = np.random.choice(a=steps, size=n_steps-1)
    rw = np.append(origin, steps).cumsum(0)

    # add outliers
    n_outliers = int(np.round(perc_outliers * n_steps, 0))
    indices = np.random.randint(0, len(rw), n_outliers)
    rw[indices] = rw[indices] + steps[indices + 1] * outlier_mult

    return rw, indices
