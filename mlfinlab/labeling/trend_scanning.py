# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Implementation of Trend-Scanning labels described in `Advances in Financial Machine Learning: Lecture 3/10
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_
"""

import pandas as pd
import numpy as np

from mlfinlab.structural_breaks.sadf import get_betas


def trend_scanning_labels(price_series: pd.Series, t_events: list = None, observation_window: int = 20,
                          look_forward: bool = True, min_sample_length: int = 5, step: int = 1) -> pd.DataFrame:
    """
    `Trend scanning <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419>`_ is both a classification and
    regression labeling technique.

    That can be used in the following ways:

    1. Classification: By taking the sign of t-value for a given observation we can set {-1, 1} labels to define the
       trends as either downward or upward.
    2. Classification: By adding a minimum t-value threshold you can generate {-1, 0, 1} labels for downward, no-trend,
       upward.
    3. The t-values can be used as sample weights in classification problems.
    4. Regression: The t-values can be used in a regression setting to determine the magnitude of the trend.

    The output of this algorithm is a DataFrame with t1 (time stamp for the farthest observation), t-value, returns for
    the trend, and bin.

    This function allows using both forward-looking and backward-looking window (use the look_forward parameter).

    :param price_series: (pd.Series) Close prices used to label the data set
    :param t_events: (list) Filtered events, array of pd.Timestamps
    :param observation_window: (int) Maximum look forward window used to get the trend value
    :param look_forward: (bool) True if using a forward-looking window, False if using a backward-looking one
    :param min_sample_length: (int) Minimum sample length used to fit regression
    :param step: (int) Optimal t-value index is searched every 'step' indices
    :return: (pd.DataFrame) Consists of t1, t-value, ret, bin (label information). t1 - label endtime, tvalue,
        ret - price change %, bin - label value based on price change sign
    """
    # pylint: disable=invalid-name, too-many-locals

    if t_events is None:
        t_events = price_series.index

    t1_array = []  # Array of label end times
    t_values_array = []  # Array of trend t-values

    for obs_id, index in enumerate(t_events):
        # Change of subset depending on looking forward or backward
        if look_forward:
            subset = price_series.loc[index:].iloc[:observation_window]  # Take t:t+L window
        else:
            subset = price_series.loc[:index].iloc[max(0, obs_id-observation_window):]  # Take t-L:t window
        if subset.shape[0] >= observation_window:
            # Loop over possible look-ahead windows to get the one which yields maximum t values for b_1 regression coef
            max_abs_t_value = -np.inf  # Maximum abs t-value of b_1 coefficient among l values
            max_t_value_index = None  # Index with maximum t-value
            max_t_value = None  # Maximum t-value signed

            # Get optimal label end time value based on regression t-statistics
            for forward_window in np.arange(min_sample_length, subset.shape[0], step):
                # Change of y_subset depending on looking forward or backward
                if look_forward:
                    y_subset = subset.iloc[:forward_window].values.reshape(-1, 1)  # y{t}:y_{t+l}
                else:
                    y_subset = subset.iloc[-forward_window:].values.reshape(-1, 1)  # y{t-l}:y_{t}

                # Array of [1, 0], [1, 1], [1, 2], ... [1, l] # b_0, b_1 coefficients
                X_subset = np.ones((y_subset.shape[0], 2))
                X_subset[:, 1] = np.arange(y_subset.shape[0])

                # Get regression coefficients estimates
                b_mean_, b_std_ = get_betas(X_subset, y_subset)
                # Check if l gives the maximum t-value among all values {0...L}
                t_beta_1 = (b_mean_[1] / np.sqrt(b_std_[1, 1]))[0]
                if abs(t_beta_1) > max_abs_t_value:
                    max_abs_t_value = abs(t_beta_1)
                    max_t_value = t_beta_1
                    max_t_value_index = forward_window

            # Store label information (t1, return)
            label_endtime_index = subset.index[max_t_value_index - 1]
            t1_array.append(label_endtime_index)
            t_values_array.append(max_t_value)

        else:
            t1_array.append(None)
            t_values_array.append(None)

    labels = pd.DataFrame({'t1': t1_array, 't_value': t_values_array}, index=t_events)
    labels.loc[:, 'ret'] = price_series.reindex(labels.t1).values / price_series.reindex(labels.index).values - 1
    labels['bin'] = labels.t_value.apply(np.sign)

    return labels
