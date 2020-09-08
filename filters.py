import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rema_filter_update(series_last, rema_last, length=14, lamb=0.5):
    # regularized ema
    alpha = 2 / (length + 1)
    rema = (rema_last + alpha * (series_last - rema_last) + lamb * (2 * rema_last - rema[2])) / (lamb + 1)
    return rema


def rema_filter(series, length:int, lamb:float):
    rema_next = series[0]
    rema = []
    for value in series:
        rema_next = rema_filter_update(
            series_last=value, rema_last=rema_next, length=length, lamb=lamb
        )
        rema.append(rema_next)
    # rema.pop(0)
    # rema[0:length] = [None] * length
    return rema


def jma_filter_update(series_last:float, e0_last:float, e1_last:float, e2_last:float, jma_last:float, length=7, phase=50, power=2):
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
    return jma_next, e0_next, e1_next, e2_next,


def jma_filter(series, length:int=7, phase:int=50, power:int=2):
    jma_next = series[0]
    e0_next = 0
    e1_next = 0
    e2_next = 0
    jma = []
    for value in series:
        jma_next, e0_next, e1_next, e2_next  = jma_filter_update(
            series_last=value, e0_last=e0_next, e1_last=e1_next,
            e2_last=e2_next, jma_last=jma_next, length=length,
            phase=phase, power=power
        )
        jma.append(jma_next)
    # jma.pop(0)
    jma[0:length] = [None] * length
    return jma


def add_filters(df, col):
    df['med_smooth5'] = df[col].rolling(window=5, center=True, min_periods=1).median()
    df['med_filter5'] = df[col].rolling(window=5, center=False, min_periods=1).median()
    df['med_filter7'] = df[col].rolling(window=7, center=False, min_periods=1).median()
    df['jma_filter3'] = jma_filter(df[col], length=3, phase=50, power=2)
    df['jma_filter7'] = jma_filter(df[col], length=7, phase=50, power=2)
    df['jma_filter9'] = jma_filter(df[col], length=9, phase=50, power=2)
    return df


def add_outlier_metrics(df, series):
    df['outlier_diff'] = abs(df['price'] - series)
    df['outlier_pct'] = abs((1-(df['price'] / series)))*100
    df['outlier_zs'] = (df['outlier_diff'] - df['outlier_diff'].mean()) / df['outlier_diff'].std(ddof=0)
    return df

