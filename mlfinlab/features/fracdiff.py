# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Fractional differentiation is a technique to make a time series stationary but also
retain as much memory as possible.  This is done by differencing by a positive real
number. Fractionally differenced series can be used as a feature in machine learning
process.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller


class FractionalDifferentiation:
    """
    FractionalDifferentiation class encapsulates the functions that can
    be used to compute fractionally differentiated series.
    """

    @staticmethod
    def get_weights(diff_amt, size):
        """
        Advances in Financial Machine Learning, Chapter 5, section 5.4.2, page 79.

        The helper function generates weights that are used to compute fractionally
        differentiated series. It computes the weights that get used in the computation
        of  fractionally differentiated series. This generates a non-terminating series
        that approaches zero asymptotically. The side effect of this function is that
        it leads to negative drift "caused by an expanding window's added weights"
        (see page 83 AFML)

        When diff_amt is real (non-integer) positive number then it preserves memory.

        The book does not discuss what should be expected if d is a negative real
        number. Conceptually (from set theory) negative d leads to set of negative
        number of elements. And that translates into a set whose elements can be
        selected more than once or as many times as one chooses (multisets with
        unbounded multiplicity) - see http://faculty.uml.edu/jpropp/msri-up12.pdf.

        :param diff_amt: (float) Differencing amount
        :param size: (int) Length of the series
        :return: (np.ndarray) Weight vector
        """

        # The algorithm below executes the iterative estimation (section 5.4.2, page 78)
        weights = [1.]  # create an empty list and initialize the first element with 1.
        for k in range(1, size):
            weights_ = -weights[-1] * (diff_amt - k + 1) / k  # compute the next weight
            weights.append(weights_)

        # Now, reverse the list, convert into a numpy column vector
        weights = np.array(weights[::-1]).reshape(-1, 1)
        return weights

    @staticmethod
    def frac_diff(series, diff_amt, thresh=0.01):
        """
        Advances in Financial Machine Learning, Chapter 5, section 5.5, page 82.

        References:
        https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
        https://wwwf.imperial.ac.uk/~ejm/M3S8/Problems/hosking81.pdf
        https://en.wikipedia.org/wiki/Fractional_calculus

        The steps are as follows:
        - Compute weights (this is a one-time exercise)
        - Iteratively apply the weights to the price series and generate output points

        This is the expanding window variant of the fracDiff algorithm
        Note 1: For thresh-1, nothing is skipped
        Note 2: diff_amt can be any positive fractional, not necessarility bounded [0, 1]

        :param series: (pd.DataFrame) A time series that needs to be differenced
        :param diff_amt: (float) Differencing amount
        :param thresh: (float) Threshold or epsilon
        :return: (pd.DataFrame) Differenced series
        """

        # 1. Compute weights for the longest series
        weights = get_weights(diff_amt, series.shape[0])

        # 2. Determine initial calculations to be skipped based on weight-loss threshold
        weights_ = np.cumsum(abs(weights))
        weights_ /= weights_[-1]
        skip = weights_[weights_ > thresh].shape[0]

        # 3. Apply weights to values
        output_df = {}
        for name in series.columns:
            series_f = series[[name]].fillna(method='ffill').dropna()
            output_df_ = pd.Series(index=series.index, dtype='float64')

            for iloc in range(skip, series_f.shape[0]):
                loc = series_f.index[iloc]

                # At this point all entries are non-NAs so no need for the following check
                # if np.isfinite(series.loc[loc, name]):
                output_df_[loc] = np.dot(weights[-(iloc + 1):, :].T, series_f.loc[:loc])[0, 0]

            output_df[name] = output_df_.copy(deep=True)
        output_df = pd.concat(output_df, axis=1)
        return output_df

    @staticmethod
    def get_weights_ffd(diff_amt, thresh, lim):
        """
        Advances in Financial Machine Learning, Chapter 5, section 5.4.2, page 83.

        The helper function generates weights that are used to compute fractionally
        differentiate dseries. It computes the weights that get used in the computation
        of fractionally differentiated series. The series is of fixed width and same
        weights (generated by this function) can be used when creating fractional
        differentiated series.
        This makes the process more efficient. But the side-effect is that the
        fractionally differentiated series is skewed and has excess kurtosis. In
        other words, it is not Gaussian any more.

        The discussion of positive and negative d is similar to that in get_weights
        (see the function get_weights)

        :param diff_amt: (float) Differencing amount
        :param thresh: (float) Threshold for minimum weight
        :param lim: (int) Maximum length of the weight vector
        :return: (np.ndarray) Weight vector
        """

        weights = [1.]
        k = 1

        # The algorithm below executes the iterativetive estimation (section 5.4.2, page 78)
        # The output weights array is of the indicated length (specified by lim)
        ctr = 0
        while True:
            # compute the next weight
            weights_ = -weights[-1] * (diff_amt - k + 1) / k

            if abs(weights_) < thresh:
                break

            weights.append(weights_)
            k += 1
            ctr += 1
            if ctr == lim - 1:  # if we have reached the size limit, exit the loop
                break

        # Now, reverse the list, convert into a numpy column vector
        weights = np.array(weights[::-1]).reshape(-1, 1)
        return weights

    @staticmethod
    def frac_diff_ffd(series, diff_amt, thresh=1e-5):
        """
        Advances in Financial Machine Learning, Chapter 5, section 5.5, page 83.

        References:

        * https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
        * https://wwwf.imperial.ac.uk/~ejm/M3S8/Problems/hosking81.pdf
        * https://en.wikipedia.org/wiki/Fractional_calculus

        The steps are as follows:

        - Compute weights (this is a one-time exercise)
        - Iteratively apply the weights to the price series and generate output points

        Constant width window (new solution)
        Note 1: thresh determines the cut-off weight for the window
        Note 2: diff_amt can be any positive fractional, not necessarity bounded [0, 1].

        :param series: (pd.DataFrame) A time series that needs to be differenced
        :param diff_amt: (float) Differencing amount
        :param thresh: (float) Threshold for minimum weight
        :return: (pd.DataFrame) A data frame of differenced series
        """

        # 1) Compute weights for the longest series
        weights = get_weights_ffd(diff_amt, thresh, series.shape[0])
        width = len(weights) - 1

        # 2) Apply weights to values
        # 2.1) Start by creating a dictionary to hold all the fractionally differenced series
        output_df = {}

        # 2.2) compute fractionally differenced series for each stock
        for name in series.columns:
            series_f = series[[name]].fillna(method='ffill').dropna()
            temp_df_ = pd.Series(index=series.index, dtype='float64')
            for iloc1 in range(width, series_f.shape[0]):
                loc0 = series_f.index[iloc1 - width]
                loc1 = series.index[iloc1]

                # At this point all entries are non-NAs, hence no need for the following check
                # if np.isfinite(series.loc[loc1, name]):
                temp_df_[loc1] = np.dot(weights.T, series_f.loc[loc0:loc1])[0, 0]

            output_df[name] = temp_df_.copy(deep=True)

        # transform the dictionary into a data frame
        output_df = pd.concat(output_df, axis=1)
        return output_df


def get_weights(diff_amt, size):
    """ This is a pass-through function """
    return FractionalDifferentiation.get_weights(diff_amt, size)


def frac_diff(series, diff_amt, thresh=0.01):
    """ This is a pass-through function """
    return FractionalDifferentiation.frac_diff(series, diff_amt, thresh)


def get_weights_ffd(diff_amt, thresh, lim):
    """ This is a pass-through function """
    return FractionalDifferentiation.get_weights_ffd(diff_amt, thresh, lim)


def frac_diff_ffd(series, diff_amt, thresh=1e-5):
    """
    Advances in Financial Machine Learning, Chapter 5, section 5.5, page 83.

    References:

    * https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
    * https://wwwf.imperial.ac.uk/~ejm/M3S8/Problems/hosking81.pdf
    * https://en.wikipedia.org/wiki/Fractional_calculus

    The steps are as follows:

    - Compute weights (this is a one-time exercise)
    - Iteratively apply the weights to the price series and generate output points

    Constant width window (new solution)
    Note 1: thresh determines the cut-off weight for the window
    Note 2: diff_amt can be any positive fractional, not necessarity bounded [0, 1].

    :param series: (pd.DataFrame) A time series that needs to be differenced
    :param diff_amt: (float) Differencing amount
    :param thresh: (float) Threshold for minimum weight
    :return: (pd.DataFrame) A data frame of differenced series
    """
    return FractionalDifferentiation.frac_diff_ffd(series, diff_amt, thresh)

def plot_min_ffd(series):
    """
    Advances in Financial Machine Learning, Chapter 5, section 5.6, page 85.

    References:

    * https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086

    This function plots the graph to find the minimum D value that passes the ADF test.

    It allows to determine d - the amount of memory that needs to be removed to achieve
    stationarity. This function covers the case of 0 < d << 1, when the original series is
    "mildly non-stationary."

    The right y-axis on the plot is the ADF statistic computed on the input series downsampled
    to a daily frequency.

    The x-axis displays the d value used to generate the series on which the ADF statistic is computed.

    The left y-axis plots the correlation between the original series (d=0) and the differentiated
    series at various d values.

    Examples on how to interpret the results of this function are available in the corresponding part
    in the book Advances in Financial Machine Learning.

    :param series: (pd.DataFrame) Dataframe that contains a 'close' column with prices to use.
    :return: (plt.AxesSubplot) A plot that can be displayed or used to obtain resulting data.
    """

    results = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])

    # Iterate through d values with 0.1 step
    for d_value in np.linspace(0, 1, 11):
        close_prices = np.log(series[['close']]).resample('1D').last()  # Downcast to daily obs
        close_prices.dropna(inplace=True)

        # Applying fractional differentiation
        differenced_series = frac_diff_ffd(close_prices, diff_amt=d_value, thresh=0.01).dropna()

        # Correlation between the original and the differentiated series
        corr = np.corrcoef(close_prices.loc[differenced_series.index, 'close'],
                           differenced_series['close'])[0, 1]
        # Applying ADF
        differenced_series = adfuller(differenced_series['close'], maxlag=1, regression='c', autolag=None)

        # Results to dataframe
        results.loc[d_value] = list(differenced_series[:4]) + [differenced_series[4]['5%']] + [corr]  # With critical value

    # Plotting
    plot = results[['adfStat', 'corr']].plot(secondary_y='adfStat', figsize=(10, 8))
    plt.axhline(results['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')

    return plot
