import numpy as np
import pandas as pd
# https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d



def mad_filter(ticks_df: pd.DataFrame, col: str='price', value_winlen: int=11,
    devations_winlen: int=333, k: int=22) -> pd.DataFrame:

    df = ticks_df[ticks_df.irregular==False].copy()
    df[col+'_median'] = df[col].rolling(value_winlen, min_periods=value_winlen, center=False).median()
    df[col+'_median_diff'] = abs(df[col] - df[col+'_median'])
    df[col+'_median_diff_median'] = df[col+'_median_diff'].rolling(devations_winlen, min_periods=value_winlen, center=False).median()
    df.loc[df[col+'_median_diff_median'] < 0.005, col+'_median_diff_median'] = 0.005  # enforce lower bound
    df.loc[df[col+'_median_diff_median'] > 0.05, col+'_median_diff_median'] = 0.05  # enforce max bound
    df['mad_outlier'] = abs(df[col] - df[col+'_median']) > (df[col+'_median_diff_median'] * k)
    df = df.dropna()
    print(df.mad_outlier.value_counts() / df.shape[0])

    return df


class MAD:
    def __init__(self, value_winlen: int=11, deviation_winlen: int=333, k: int=22):
        self.value_winlen = value_winlen
        self.deviation_winlen = deviation_winlen
        self.k = k
        self.values = []
        self.deviations = []
        self.status = 'mad_warmup'

    def update(self, next_value):
        self.values.append(next_value)
        self.values = self.values[-self.value_winlen:]  # only keep winlen
        self.value_median = np.median(self.values)
        self.value_median_diff = next_value - self.value_median
        # self.value_median_pct_change = self.value_median_diff / self.value_median
        self.deviations.append(self.value_median_diff)
        self.deviations = self.deviations[-self.deviation_winlen:]  # only keep winlen
        self.deviations_median = np.median(self.deviations)
        self.deviations_median = 0.005 if self.deviations_median < 0.005 else self.deviations_median  # enforce lower limit
        self.deviations_median = 0.05 if self.deviations_median > 0.05 else self.deviations_median  # enforce upper limit
        # final tick status logic
        if len(self.values) < self.value_winlen:
            self.status = 'mad_warmup'
        elif abs(self.value_median_diff) > (self.deviations_median * self.k):
            self.status = 'mad_outlier'    
        else:
            self.status = 'mad_clean'

        return self.value_median


class JMA:
    def __init__(self, start_value: float, winlen: int=10, power: int=1, phase: float=0.0):
        self.state = jma_starting_state(start_value)
        self.winlen = winlen
        self.power = power
        self.phase = phase

    def update(self, next_value):
        self.state = jma_filter_update(
            value=next_value,
            state=self.state,
            winlen=self.winlen,
            power=self.power,
            phase=self.phase
            )
        return self.state['jma']


def jma_starting_state(start_value: float) -> dict:
    return {
        'e0': start_value,
        'e1': 0.0,
        'e2': 0.0,
        'jma': start_value,
        }


def jma_filter_update(value: float, state: dict, winlen: int, power: float, phase: float) -> dict:
    if phase < -100:
        phase_ratio = 0.5
    elif phase > 100:
        phase_ratio = 2.5
    else:
        phase_ratio = phase / (100 + 1.5)
    beta = 0.45 * (winlen - 1) / (0.45 * (winlen - 1) + 2)
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


def jma_rolling_filter(series: pd.Series, winlen: int, power: float, phase: float) -> list:
    jma = []
    state = jma_starting_state(start_value=series.values[0])
    for value in series:
        state = jma_filter_update(value, state, winlen, power, phase)
        jma.append(state['jma'])

    jma[0:(winlen-1)] = [None] * (winlen-1)
    return jma


def jma_expanding_filter(series: pd.Series, winlen: int, power: float, phase: float) -> list:
    
    if winlen < 1:
        raise ValueError('winlen parameter must be >= 1')
    running_jma = jma_rolling_filter(series, winlen, power, phase)
    expanding_jma = []
    for winlen_exp in list(range(1, winlen)):
        jma = jma_rolling_filter(series[0:winlen_exp], winlen_exp, power, phase)
        expanding_jma.append(jma[winlen_exp-1])
    
    running_jma[0:(winlen-1)] = expanding_jma
    return running_jma


def jma_filter_df(df: pd.DataFrame, col: str, winlen: int, power: float, phase: float=0, expand: bool=False) -> pd.DataFrame:
    if expand:
        df.loc[:, col+'_jma'] = jma_expanding_filter(df[col], winlen, power, phase)
    else:
        df.loc[:, col+'_jma'] = jma_rolling_filter(df[col], winlen, power, phase)
    return df


def rema_filter_update(series_last: float, rema_last: float, winlen: int=14, lamb: float=0.5) -> float:
    # regularized ema
    alpha = 2 / (winlen + 1)
    rema = (rema_last + alpha * (series_last - rema_last) + 
        lamb * (2 * rema_last - rema[2])) / (lamb + 1)
    return rema


def rema_filter(series: pd.Series, winlen: int, lamb: float) -> list:
    rema_next = series.values[0]
    rema = []
    for value in series:
        rema_next = rema_filter_update(
            series_last=value, rema_last=rema_next, winlen=winlen, lamb=lamb
        )
        rema.append(rema_next)
    return rema


def supersmoother(x: list, n: int=10) -> np.ndarray:
    from math import exp, cos, radians
    assert (n > 0) and (n < len(x))
    a = exp(-1.414 * 3.14159 / n)
    b = cos(radians(1.414 * 180 / n))
    c2 = b
    c3 = -a * a
    c1 = 1 - c2 - c3
    ss = np.zeros(len(x))
    for i in range(3, len(x)):
        ss[i] = c1 * (x[i] + x[i-1]) / 2 + c2 * ss[i-1] + c3 * ss[i-2]
    return ss


def tick_df_fix(ticks_df: pd.DataFrame) -> pd.DataFrame:
    t1df = ticks_df.set_index('sip_dt').tz_localize('UTC').tz_convert('America/New_York')
    t1df = t1df.between_time('9:00', '16:00').reset_index()
    t1df = t1df.drop(columns=['exchange_dt'])
    return t1df.rename(columns={"sip_dt": "date_time"})
