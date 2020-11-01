import numpy as np
import pandas as pd


def rema_filter_update(series_last: float, rema_last: float, length=14, lamb=0.5) -> float:
    # regularized ema
    alpha = 2 / (length + 1)
    rema = (rema_last + alpha * (series_last - rema_last) + 
        lamb * (2 * rema_last - rema[2])) / (lamb + 1)
    return rema


def rema_filter(series: pd.Series, length: int, lamb: float) -> list:
    rema_next = series.values[0]
    rema = []
    for value in series:
        rema_next = rema_filter_update(
            series_last=value, rema_last=rema_next, length=length, lamb=lamb
        )
        rema.append(rema_next)
    # rema.pop(0)
    # rema[0:length] = [None] * length
    return rema


def jma_filter_update(series_last: float, e0_last: float, e1_last: float,
    e2_last: float, jma_last: float, length: int=7, phase: int=50, power: int=2):
    if phase < -100:
        phase_ratio = 0.5
    elif phase > 100:
        phase_ratio = 2.5
    else:    
        phase_ratio = phase / (100 + 1.5)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = pow(beta, power)
    e0_next = (1 - alpha) * series_last + alpha * e0_last
    e1_next = (series_last - e0_next) * (1 - beta) + beta * e1_last
    e2_next = (e0_next + phase_ratio * e1_next - jma_last) * pow(1 - alpha, 2) + pow(alpha, 2) * e2_last
    jma_next = e2_next + jma_last
    return jma_next, e0_next, e1_next, e2_next


def jma_rolling_filter(series: pd.Series, length: int=7, phase: int=50, power: int=2) -> list:
    e0_next = 0
    e1_next = 0
    e2_next = 0
    jma_next = series.values[0]
    jma = []
    for value in series:
        jma_next, e0_next, e1_next, e2_next  = jma_filter_update(
            series_last=value, e0_last=e0_next, e1_last=e1_next,
            e2_last=e2_next, jma_last=jma_next, length=length,
            phase=phase, power=power
        )
        jma.append(jma_next)

    jma[0:(length-1)] = [None] * (length-1)
    return jma


def jma_expanding_filter(series: pd.Series, length: int=7, phase: int=50, power: int=2) -> pd.DataFrame:
    if length < 1:
        raise ValueError('length parameter must be >= 1')
    running_jma = jma_rolling_filter(series, length, phase, power)
    expanding_jma = []
    for length_exp in list(range(1, length)):
        jma = jma_rolling_filter(series[0:length_exp], length_exp, phase, power)
        expanding_jma.append(jma[length_exp-1])
    
    running_jma[0:(length-1)] = expanding_jma
    return running_jma


def jma_filter_df(df: pd.DataFrame, col: str, expand: bool=True, length: int=7, phase: int=50, power: int=2) -> pd.DataFrame:
    if expand:
        df.loc[:, col+'_jma'] = jma_expanding_filter(df[col], length, phase, power)
    else:
        df.loc[:, col+'_jma'] = jma_rolling_filter(df[col], length, phase, power)
    return df


def add_multiple_jma_df(df: pd.DataFrame, col: str, lengths: list) -> pd.DataFrame:
    df = df.copy()
    for length in lengths:
        # compute rolling std
        df.loc[:, col+'_'+str(length)+'std'] = df[col].rolling(window=length, min_periods=0).std()
        # compute jma filter
        df.loc[:, col+'_jma_'+str(length)] = jma_expanding_filter(df[col], length=length, phase=50, power=2)
        # compute value - jma filter diff
        df.loc[:, col+'_jma_'+str(length)+'_diff'] = df[col] - df[col+'_jma_'+str(length)]
        # compute diff mean
        df.loc[:, col+'_jma_'+str(length)+'_diff_mean'] = abs(df[col+'_jma_'+str(length)+'_diff']).rolling(window=length, min_periods=0).mean()
        # compute diff median
        df.loc[:, col+'_jma_'+str(length)+'_diff_median'] = abs(df[col+'_jma_'+str(length)+'_diff']).rolling(window=length, min_periods=0).median()
    return df


def median_outlier_filter(df: pd.DataFrame, col: str='price', window: int=5, zthresh: int=10) -> pd.DataFrame:
    # https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d
    df['filter'] = df[col].rolling(window, center=False, min_periods=1).median()
    df['filter_diff'] = abs(df[col] - df['filter'])
    df['filter_zs'] = (df['filter_diff'] - df['filter_diff'].mean()) / df['filter_diff'].std(ddof=0)
    return df.loc[df.filter_zs < zthresh].reset_index(drop=True)




import sympy
cosd = lambda x : sympy.cos( sympy.mpmath.radians(x) )


def supersmoother(x: list, n: int=10) -> list:
    a = exp(-1.414 * 3.14159 / n)
    b = 2 * a * cosd(1.414 * 180 / n)
    c2 = b
    c3 = -a * a
    c1 = 1 - c2 - c3
    # @assert n<size(x,1) && n>0 "Argument n out of bounds."
    super = np.zeros(len(x))
     # @inbounds for i = 3:length(x)
    for i in range(3, len(x)):
        super[i] = c1 * (x[i] + x[i-1]) / 2 + c2 * super[i-1] + c3 * super[i-2]

    return super
