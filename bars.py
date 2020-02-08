import os
import datetime
import requests
import scipy.stats as stats
import numpy as np
import pandas as pd


def get_outliers(ts,  multiple=6):
    ts['price_pct_change'] = ts['price'].pct_change() * 100
    toss_thresh = ts['price_pct_change'].std() * multiple
    outliers_idx = ts['price_pct_change'].abs() > toss_thresh
    # ts['outliers_idx'] = outliers_idx
    # return ts[-outliers_idx]
    return outliers_idx


def epoch_to_dt(ts):
    ts['date_time'] = pd.to_datetime(ts['epoch'], utc=True, unit='ns')
    # ts['date_time_nyc'] = pd.to_datetime(ts['epoch'], utc=True, unit='ns').dt.tz_convert('US/Eastern') #'America/New_York'
    # ts['date_time_nyc'] = ts['date_time'].dt.tz_convert('US/Eastern')
    return ts


def trunc_timestamp(ts, trunc_list=None, add_date_time=False):
    if add_date_time:
        ts = epoch_to_dt(ts)
    # ts['epoch_ns'] = ts.epoch.floordiv(10 ** 1)
    if 'micro' in trunc_list:
        ts['epoch_micro'] = ts['epoch'].floordiv(10 ** 3)
    if 'ms' in trunc_list:
        ts['epoch_ms'] = ts['epoch'].floordiv(10 ** 6)
        if  add_date_time:
            ts['date_time_ms'] = ts['date_time'].values.astype('<M8[ms]')
    if 'cs' in trunc_list:
        ts['epoch_cs'] = ts['epoch'].floordiv(10 ** 7)
    if 'ds' in trunc_list:
        ts['epoch_ds'] = ts['epoch'].floordiv(10 ** 8)
    if 'sec' in trunc_list:
        ts['epoch_sec'] = ts['epoch'].floordiv(10 ** 9)
        if add_date_time:
            ts['date_time_sec'] = ts['date_time'].values.astype('<M8[s]')
    if 'min' in trunc_list:
        ts['epoch_min'] = ts['epoch'].floordiv((10 ** 9) * 60)
        if add_date_time:
            ts['date_time_min'] = ts['date_time'].values.astype('<M8[m]')
    if 'min10' in trunc_list:
        ts['epoch_min10'] = ts['epoch'].floordiv((10 ** 9) * 60 * 10)
    if 'min15' in trunc_list:
        ts['epoch_min15'] = ts['epoch'].floordiv((10 ** 9) * 60 * 15)
    if 'min30' in trunc_list:
        ts['epoch_min30'] = ts['epoch'].floordiv((10 ** 9) * 60 * 30)
    if 'hour' in trunc_list:
        ts['epoch_hour'] = ts['epoch'].floordiv((10 ** 9) * 60 * 60)
        if add_date_time:
            ts['date_time_hour'] = ts['date_time'].values.astype('<M8[h]')

    return ts


def ts_groupby(df, column='epoch_cs'):
    ts = df.copy()
    groups = ts.groupby(column, as_index=False, squeeze=True, ).agg({'price': ['count', 'mean'], 'volume':'sum'})
    groups.columns = ['_'.join(tup).rstrip('_') for tup in groups.columns.values]
    return groups


def rolling_decay_vwap(ts, window_len=7, decay='none'):
    nrows = list(range(len(ts)))
    vwap = []
    for nrow in nrows:
        if nrow >= window_len:
            price = ts['price'][nrow - window_len:nrow].values
            volume = ts['volume'][nrow - window_len:nrow].values
            if decay == 'exp':
                exp_decay = stats.expon.pdf(x=list(range(0, window_len)))
                weight = volume * exp_decay
            if decay == 'linear':
                linear_decay = np.array(range(1, window_len+1)) / window_len
                weight = volume * linear_decay
            if decay == 'none':
                weight = volume
            tmp = (price * weight).sum() / weight.sum()
            vwap.append(tmp)
        else:
            vwap.append(None)
    return vwap


def tick_rule(first_price, second_price, last_side=0):
    try:
        diff = first_price - second_price
    except:
        diff = None
    if diff > 0.0:
        side = 1
    elif diff < 0.0:
        side = -1
    elif diff == 0.0:
        side = last_side
    else:
        side = 0
    return side


def get_next_renko_thresh(thresh_renko, last_return):
    if last_return >= 0:
        thresh_renko_bull = thresh_renko
        thresh_renko_bear = -thresh_renko * 2
    elif last_return < 0:
        thresh_renko_bull = thresh_renko * 2
        thresh_renko_bear = -thresh_renko
    return thresh_renko_bull, thresh_renko_bear


def reset_state(thresh={}):
    state = {}    
    state['thresh'] = thresh
    # accululators
    state['duration_ns'] = 0
    state['price_min'] = 10 ** 5
    state['price_max'] = 0
    state['price_range'] = 0
    state['bar_return'] = 0
    state['tick_count'] = 0
    state['volume_sum'] = 0
    state['dollar_sum'] = 0
    state['tick_imbalance'] = 0
    state['volume_imbalance'] = 0
    state['dollar_imbalance'] = 0
    state['tick_imbalance_max'] = 0
    state['volume_imbalance_max'] = 0
    state['dollar_imbalance_max'] = 0
    state['tick_run'] = 0
    state['volume_run'] = 0
    state['dollar_run'] = 0
    state['tick_run_max'] = 0
    state['volume_run_max'] = 0
    state['dollar_run_max'] = 0
    # copy of tick events
    state['trades'] = {}
    state['trades']['epoch'] = []
    state['trades']['price'] = []
    state['trades']['volume'] = []
    state['trades']['side'] = []
    # trigger status
    state['next_bar'] = 'waiting'
    return state


def weighted_kernel_density_1d(values, weights, bw='silverman', plot=False):
    from statsmodels.nonparametric.kde import KDEUnivariate
    # “scott” - 1.059 * A * nobs ** (-1/5.), where A is min(std(X),IQR/1.34)
    # “silverman” - .9 * A * nobs ** (-1/5.), where A is min(std(X),IQR/1.34)
    # "normal_reference" - C * A * nobs ** (-1/5.), where C is
    kden= KDEUnivariate(values)
    kden.fit(weights=weights, bw=bw, fft=False)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(kden.support, [kden.evaluate(xi) for xi in kden.support], 'o-')
    return kden


def quantile_from_kdensity(kden, quantile=0.5):
    return kden.support[kde1.cdf >= quantile][0]


def wkde1d(state, new_bar):
    
    try:
        wkde = weighted_kernel_density_1d(
                values=np.array(state['trades']['price']), 
                weights=np.array(state['trades']['volume'])
                )
        # new_bar['wkde'] = wkde
        new_bar['kd_10'] = quantile_from_kdensity(new_bar['kden'], quantile=0.1)
        new_bar['kd_50'] = quantile_from_kdensity(new_bar['kden'], quantile=0.5)
        new_bar['kd_90'] = quantile_from_kdensity(new_bar['kden'], quantile=0.9)
    except:
        new_bar['wkde'] = None
        new_bar['kd_10'] = None
        new_bar['kd_50'] = None
        new_bar['kd_90'] = None

    return new_bar


def save_bar(state, wkde=False):
    new_bar = {}
    if state['tick_count'] == 0:
        return new_bar
    new_bar['bar_trigger'] = state['next_bar']
    new_bar['first_tick_at'] = pd.to_datetime(state['trades']['epoch'][0], utc=True, unit='ns')
    new_bar['last_tick_at'] = pd.to_datetime(state['trades']['epoch'][-1], utc=True, unit='ns')
    new_bar['duration_ns'] = state['duration_ns']
    new_bar['price_open'] = state['trades']['price'][0]
    new_bar['price_close'] = state['trades']['price'][-1]
    new_bar['price_low'] = state['price_min']
    new_bar['price_high'] = state['price_max']
    new_bar['price_vwap'] = (np.array(state['trades']['price']) * np.array(state['trades']['volume'])).sum() / np.array(state['trades']['volume']).sum()
    new_bar['price_std'] = np.array(state['trades']['price']).std()
    new_bar['price_q10'] = np.quantile(state['trades']['price'], q=0.1)
    new_bar['price_q50'] = np.quantile(state['trades']['price'], q=0.5)
    new_bar['price_q90'] = np.quantile(state['trades']['price'], q=0.9)
    new_bar['price_range'] = state['price_range']
    new_bar['bar_return'] = state['bar_return']
    new_bar['tick_count'] = state['tick_count']
    new_bar['volume_sum'] = state['volume_sum']
    new_bar['dollar_sum'] = state['dollar_sum']
    new_bar['tick_imbalance'] = state['tick_imbalance']
    new_bar['volume_imbalance'] = state['volume_imbalance']
    new_bar['dollar_imbalance'] = state['dollar_imbalance']
    new_bar['tick_imbalance_max'] = state['tick_imbalance_max']
    new_bar['volume_imbalance_max'] = state['volume_imbalance_max']
    new_bar['dollar_imbalance_max'] = state['dollar_imbalance_max']
    new_bar['tick_run_max'] = state['tick_run_max']
    new_bar['volume_run_max'] = state['volume_run_max']
    new_bar['dollar_run_max'] = state['dollar_run_max']
    new_bar = wkde1d(state, new_bar) if wkde else new_bar
    return new_bar


def bar_thresh(state, last_bar_return=0):
    if 'duration_ns' in state['thresh'] and state['duration_ns'] > state['thresh']['duration_ns']:
        state['next_bar'] = 'duration'
    if 'ticks' in state['thresh'] and state['tick_count'] >= state['thresh']['ticks']:
        state['next_bar'] = 'tick_coumt'
    if 'volume' in state['thresh'] and state['volume_sum'] >= state['thresh']['volume']:
        state['next_bar'] = 'volume_sum'
    if 'dollar' in state['thresh'] and state['dollar_sum'] >= state['thresh']['dollar']:
        state['next_bar'] = 'dollar_sum'
    if 'tick_imbalance' in state['thresh'] and abs(state['tick_imbalance']) >= state['thresh']['tick_imbalance']:
        state['next_bar'] = 'tick_imbalance'
    if 'volume_imbalance' in state['thresh'] and abs(state['volume_imbalance']) >= state['thresh']['volume_imbalance']:
        state['next_bar'] = 'volumne_imbalance'        
    if 'dollar_imbalance' in state['thresh'] and abs(state['dollar_imbalance']) >= state['thresh']['dollar_imbalance']:
        state['next_bar'] = 'dollar_imbalence'
    if 'price_range' in state['thresh'] and state['price_range'] >= state['thresh']['price_range']:
        state['next_bar'] = 'price_range'
    if 'return' in state['thresh'] and abs(state['bar_return']) >= state['thresh']['return']:
        state['next_bar'] = 'bar_return'
    if 'renko' in state['thresh']:
        try:
            state['thresh']['renko_bull'], state['thresh']['renko_bear'] = get_next_renko_thresh(
                thresh_renko=state['thresh']['renko'], 
                last_return=last_bar_return
                )
        except:
            state['thresh']['renko_bull'] = state['thresh']['renko']
            state['thresh']['renko_bear'] = -state['thresh']['renko']

        if state['bar_return'] >= state['thresh']['renko_bull']:
            state['next_bar'] = 'renko'
        if state['bar_return'] < state['thresh']['renko_bear']:
            state['next_bar'] = 'renko'
    if 'tick_run' in state['thresh'] and state['tick_run'] >= state['thresh']['tick_run']:
        state['next_bar'] = 'tick_run'
    if 'volume_run' in state['thresh'] and state['volume_run'] >= state['thresh']['volume_run']:
        state['next_bar'] = 'volume_run'
    if 'dollar_run' in state['thresh'] and state['dollar_run'] >= state['thresh']['dollar_run']:
        state['next_bar'] = 'dollar_run'

    return state


def imbalance_runs(state):
    
    if len(state['trades']['side']) >= 2:
        
        if state['trades']['side'][-1] == state['trades']['side'][-2]:
            state['tick_run'] += 1        
            state['volume_run'] += state['trades']['volume'][-1]
            state['dollar_run'] += state['trades']['price'][-1] * state['trades']['volume'][-1]
        else:
            state['tick_run'] = 0
            state['volume_run'] = 0
            state['dollar_run'] = 0

    state['tick_run_max'] = state['tick_run'] if state['tick_run'] > state['tick_run_max'] else state['tick_run_max']
    state['volume_run_max'] = state['volume_run'] if state['volume_run'] > state['volume_run_max'] else state['volume_run_max']
    state['dollar_run_max'] = state['dollar_run'] if state['dollar_run'] > state['dollar_run_max'] else state['dollar_run_max']

    return state


def imbalance_net(state):
    state['tick_imbalance'] += state['trades']['side'][-1]
    state['volume_imbalance'] += state['trades']['side'][-1] * state['trades']['volume'][-1]
    state['dollar_imbalance'] += state['trades']['side'][-1] * state['trades']['volume'][-1] * state['trades']['price'][-1]

    state['tick_imbalance_max'] = state['tick_imbalance'] if state['tick_imbalance'] > state['tick_imbalance_max'] else state['tick_imbalance_max']
    state['volume_imbalance_max'] = state['volume_imbalance'] if state['volume_imbalance'] > state['volume_imbalance_max'] else state['volume_imbalance_max']
    state['dollar_imbalance_max'] = state['dollar_imbalance'] if state['dollar_imbalance'] > state['dollar_imbalance_max'] else state['dollar_imbalance_max']    
    return state


def update_bar(tick, output_bars, state, thresh={}):
    if tick['volume'] <= 0:
        print('dropping zero volume tick')
        print(tick)
        return output_bars, state

    state['trades']['epoch'].append(tick['epoch'])
    state['trades']['price'].append(tick['price'])
    state['trades']['volume'].append(tick['volume'])

    if len(state['trades']['price']) >= 2:
        tick_side = tick_rule(first_price=state['trades']['price'][-1], 
                            second_price=state['trades']['price'][-2], 
                            last_side=state['trades']['side'][-1])
    else: 
        tick_side = 0
    state['trades']['side'].append(tick_side)
    
    state = imbalance_net(state)

    state = imbalance_runs(state)
    # state['now_diff'] = int(datetime.datetime.utcnow().timestamp() * 1000000000) - state['trades']['epoch'][0]
    state['duration_ns'] = tick['epoch'] - state['trades']['epoch'][0]
    state['tick_count'] += 1
    state['volume_sum'] += tick['volume']
    state['dollar_sum'] += tick['price'] * tick['volume']
    state['price_min'] = tick['price'] if tick['price'] < state['price_min'] else state['price_min']
    state['price_max'] = tick['price'] if tick['price'] > state['price_max'] else state['price_max']
    state['price_range'] = state['price_max'] - state['price_min']
    state['bar_return'] = tick['price'] - state['trades']['price'][0]
    
    last_bar_side = output_bars[-1]['bar_return'] if len(output_bars) > 0 else 0
    state = bar_thresh(state, last_bar_return=last_bar_side)

    if state['next_bar'] != 'waiting':
        print('new bar: ', state['next_bar'])
        # save new bar
        new_bar = save_bar(state)
        output_bars.append(new_bar)
        # reset counter vars
        state = reset_state(thresh)

    return output_bars, state


def build_bars(ts, thresh={}, as_df=True):
    
    state = reset_state(thresh)
    output_bars = []
    nrow = 0
    while nrow < len(ts):
        tick = {
            'epoch': ts['epoch'].values[nrow],
            'price': ts['price'].values[nrow],
            'volume': ts['volume'].values[nrow]
        }
        output_bars, state = update_bar(tick, output_bars, state, thresh)
        nrow += 1

    if as_df:
        output_bars = pd.DataFrame(output_bars)

    return output_bars, state


def time_bars(ts, date, freq='15min', as_df=True):
    
    ts = epoch_to_dt(ts)
    start_date = datetime.datetime.strptime(date, '%Y-%m-%d')
    end_date = start_date + datetime.timedelta(days=1)
    dr = pd.date_range(start=start_date, end=end_date, freq='5min', tz='utc', closed=None)
    new_bars = []
    
    for i in list(range(len(dr)-2)):
        ticks = ts[(ts.date_time >= dr[i]) & (ts.date_time < dr[i+1])]
        _, state = build_bars(ts=ticks, thresh={})
        bar = save_bar(state)
        bar['time_open'] = dr[i]
        new_bars.append(bar)

    new_bars = pd.DataFrame(new_bars) if as_df is True else new_bars
    return new_bars
