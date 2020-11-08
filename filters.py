import math
import numpy as np
import pandas as pd


def median_outlier_filter(df: pd.DataFrame, col: str='price', window: int=5, zthresh: int=10) -> pd.DataFrame:
    # https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d
    df['filter'] = df[col].rolling(window, center=False, min_periods=window).median()
    df['filter_diff'] = abs(df[col] - df['filter'])
    df['filter_zs'] = (df['filter_diff'] - df['filter_diff'].mean()) / df['filter_diff'].std(ddof=0)
    return df.loc[df.filter_zs < zthresh].reset_index(drop=True)


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


def jma_filter_update(value: float, state: dict, length: int=7, phase: int=0, power: int=2) -> dict:

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

    new_state = {
        'e0': e0_next, 
        'e1': e1_next, 
        'e2': e2_next,
        'jma': jma_next,
        }
    return new_state


def jma_rolling_filter(series: pd.Series, length: int=7, phase: int=50, power: int=2) -> list:
    state = {'e0': 0, 'e1': 0, 'e2': 0, 'jma': series.values[0]}
    jma = []
    for value in series:
        state = jma_filter_update(value=value, state=state, length=length, phase=phase, power=power)
        jma.append(state['jma'])

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


def add_volitiliy_features(df, col: str, length: int) -> pd.DataFrame:   
    df.loc[:, col+'_'+str(length)+'_std'] = df[col].rolling(window=length, min_periods=0).std()
    return df


def add_filter_features(df: pd.DataFrame, col: str, length: int=10, power: float=1.0) -> pd.DataFrame:
    # compute jma filter
    df.loc[:, col+'_jma_'+str(length)] = jma_expanding_filter(df[col], length=length, power=power)
    # jma diff(n)
    df.loc[:, col+'_jma_'+str(length)+'_diff1'] = df.loc[:, col+'_jma_'+str(length)].diff(1)
    df.loc[:, col+'_jma_'+str(length)+'_diff3'] = df.loc[:, col+'_jma_'+str(length)].diff(3)
    df.loc[:, col+'_jma_'+str(length)+'_diff6'] = df.loc[:, col+'_jma_'+str(length)].diff(6)
    df.loc[:, col+'_jma_'+str(length)+'_diff9'] = df.loc[:, col+'_jma_'+str(length)].diff(9)
    # compute value - jma filter residual
    df.loc[:, col+'_jma_'+str(length)+'_resid'] = df[col] - df[col+'_jma_'+str(length)]
    # compute resid abs mean
    df.loc[:, col+'_jma_'+str(length)+'_resid_mean'] = abs(df[col+'_jma_'+str(length)+'_resid']).rolling(window=length, min_periods=0).mean()
    # compute resid abs median
    df.loc[:, col+'_jma_'+str(length)+'_resid_median'] = abs(df[col+'_jma_'+str(length)+'_resid']).rolling(window=length, min_periods=0).median()
    # compute resid abs jma
    df.loc[:, col+'_jma_'+str(length)+'_resid_jma'] = jma_expanding_filter(abs(df[col+'_jma_'+str(length)+'_resid']), length=length, phase=50, power=2)
    return df


def add_bands(df: pd.DataFrame, base_col: str, vol_col: str, multipler: int=2):
    df.loc[:, base_col+'_upper'] = df[base_col] + df[vol_col] * multipler
    df.loc[:, base_col+'_lower'] = df[base_col] - df[vol_col] * multipler
    return df


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
