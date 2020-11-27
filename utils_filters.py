import math
import numpy as np
import pandas as pd


def jma_filter_update(value: float, state: dict, length: int, power: int, phase: int=0) -> dict:
    if phase < -100:
        phase_ratio = 0.5
    elif phase > 100:
        phase_ratio = 2.5
    else:
        phase_ratio = phase / (100 + 1.5)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = pow(beta, power)
    e0_next = (1 - alpha) * value + alpha * state['e0']
    e1_next = (value - e0_next) * (1 - beta) + beta * state['e1']
    e2_next = (e0_next + phase_ratio * e1_next - state['jma']) * pow(1 - alpha, 2) + pow(alpha, 2) * state['e2']
    jma_next = e2_next + state['jma']
    state_next = {
        'e0': e0_next,
        'e1': e1_next,
        'e2': e2_next,
        'jma': jma_next,
        }
    return state_next


def jma_rolling_filter(series: pd.Series, length: int, power: int, phase: int=0) -> list:
    state = {'e0': 0, 'e1': 0, 'e2': 0, 'jma': series.values[0]}
    jma = []
    for value in series:
        state = jma_filter_update(value, state, length, power, phase)
        jma.append(state['jma'])

    jma[0:(length-1)] = [None] * (length-1)
    return jma


def jma_expanding_filter(series: pd.Series, length: int, power: int, phase: int=0) -> pd.DataFrame:
    
    if length < 1:
        raise ValueError('length parameter must be >= 1')
    running_jma = jma_rolling_filter(series, length, power, phase)
    expanding_jma = []
    for length_exp in list(range(1, length)):
        jma = jma_rolling_filter(series[0:length_exp], length_exp, power, phase)
        expanding_jma.append(jma[length_exp-1])
    
    running_jma[0:(length-1)] = expanding_jma
    return running_jma


def jma_filter_df(df: pd.DataFrame, col: str, length: int, power: float, phase: int=0, expand: bool=False) -> pd.DataFrame:
    if expand:
        df.loc[:, col+'_jma'] = jma_expanding_filter(df[col], length, power, phase)
    else:
        df.loc[:, col+'_jma'] = jma_rolling_filter(df[col], length, power, phase)
    return df


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
    return rema


def supersmoother(x: list, n: int=10) -> np.ndarray:
    assert (n > 0) and (n < len(x))
    a = math.exp(-1.414 * 3.14159 / n)
    b = math.cos(math.radians(1.414 * 180 / n))
    c2 = b
    c3 = -a * a
    c1 = 1 - c2 - c3
    ss = np.zeros(len(x))
    for i in range(3, len(x)):
        ss[i] = c1 * (x[i] + x[i-1]) / 2 + c2 * ss[i-1] + c3 * ss[i-2]
    return ss


def median_outlier_filter(df: pd.DataFrame, col: str='price', window: int=5, zthresh: int=10) -> pd.DataFrame:
    # https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d
    df['filter'] = df[col].rolling(window, center=False, min_periods=window).median()
    df['filter_diff'] = abs(df[col] - df['filter'])
    df['filter_zs'] = (df['filter_diff'] - df['filter_diff'].mean()) / df['filter_diff'].std(ddof=0)
    return df.loc[df.filter_zs < zthresh].reset_index(drop=True)
