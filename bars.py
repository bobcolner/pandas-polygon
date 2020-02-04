import os
import glob
import datetime
import requests
import scipy.stats as stats
import numpy as np
import pandas as pd


def read_matching_files(glob_string, format='csv'):
    if format == 'csv':
        return pd.concat(map(pd.read_csv, glob.glob(os.path.join('', glob_string))), ignore_index=True)
    elif format == 'parquet':
        return pd.concat(map(pd.read_parquet, glob.glob(os.path.join('', glob_string))), ignore_index=True)
    elif format == 'feather':
        return pd.concat(map(pd.read_feather, glob.glob(os.path.join('', glob_string))), ignore_index=True)


def get_outliers(ts,  multiple=6):
    ts['price_pct_change'] = ts['price'].pct_change() * 100
    toss_thresh = ts['price_pct_change'].std() * multiple
    outliers_idx = ts['price_pct_change'].abs() > toss_thresh
    # ts['outliers_idx'] = outliers_idx
    # return ts[-outliers_idx]
    return outliers_idx


def sign_trades(ts):

    ts['price_pct_change'] = ts.price.pct_change() * 100

    tick_side = []
    tick_side.append(1)
    nrows = list(range(len(ts)))    
    for nrow in nrows:
        if ts['price_pct_change'][nrow] > 0.0:
            tick_side.append(1)
        elif ts['price_pct_change'][nrow] < 0.0:
            tick_side.append(-1)
        elif ts['price_pct_change'][nrow] == 0.0:
            tick_side.append(tick_side[-1])

    ts['side'] = tick_side
    return ts


def epoch_to_datetime(ts, column='epoch'):
    ts['date_time'] = pd.to_datetime(ts[column], utc=True, unit='ns')
    # ts['date_time_nyc'] = ts['date_time'].tz_convert('America/New_York')
    return ts


def epoch_to_dt(epoch):
    return pd.to_datetime(epoch, utc=True, unit='ns') # tz_convert('America/New_York')


def ts_epoch_to_dt(ts):
    ts['date_time'] = epoch_to_dt(ts['epoch'].values)
    return ts


def trunc_timestamp(ts, trunc_list=None, add_date_time=False):
    if add_date_time:
        ts['date_time'] = pd.to_datetime(ts['epoch'].values, utc=True, unit='ns')
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


def find_bar_params(ts, num_bars=100):
    d = {}
    d['volume_thresh'] = round(sum(ts['volume']) / num_bars)
    d['tick_thresh'] = round(len(ts) / num_bars)
    d['dollar_thresh'] = round((np.mean(ts['price']) * sum(ts['volume'])) / num_bars)
    d['minute_thresh'] = round((6.5 / num_bars) * 60)
    return d


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


def update_price_stats(price, price_min, price_max):
    if price < price_min:
        price_min = price
    if price > price_max:
        price_max = price
    price_range = price_max - price_min
    return price_range, price_min, price_max


def get_next_renko_thresh(thresh_renko, last_return):
    if last_return >= 0:
        thresh_renko_bull = thresh_renko
        thresh_renko_bear = -thresh_renko * 2
    elif last_return < 0:
        thresh_renko_bull = thresh_renko * 2
        thresh_renko_bear = -thresh_renko
    return thresh_renko_bull, thresh_renko_bear


def reset_state():
    state = {}    
    state['thresh_duration_ns'] = (10 ** 9) * 60 * 10
    state['thresh_ticks'] = 10 ** 3
    state['thresh_volume'] = 10 ** 5
    state['thresh_dollar'] = 10 ** 6
    state['thresh_tick_imbalance'] = 10 ** 3
    state['thresh_volume_imbalance'] = 10 ** 4
    state['thresh_dollar_imbalance']  = 10 ** 5 
    state['thresh_price_range'] = 0.3
    state['thresh_return'] = 0.2
    state['thresh_renko'] = 0.1

    state['duration_ns'] = 0
    state['price_min'] = 10 ** 5
    state['price_max'] = 0
    state['price_range'] = 0
    state['bar_return'] = 0
    state['ticks'] = 0
    state['volume'] = 0
    state['dollar'] = 0
    state['tick_imbalance'] = 0
    state['volume_imbalance'] = 0
    state['dollar_imbalance'] = 0
    state['trades'] = {}
    state['trades']['epoch'] = []
    state['trades']['price'] = []
    state['trades']['volume'] = []
    state['trades']['side'] = []
    state['next_bar'] = 'waiting'
    return state


def save_bar(state):
    new_bar = {}
    new_bar['bar_trigger'] = state['next_bar']
    new_bar['open_time'] = state['trades']['epoch'][0]
    new_bar['close_time'] = state['trades']['epoch'][-1]
    new_bar['duration_ns'] = state['duration_ns']
    new_bar['price_open'] = state['trades']['price'][0]
    new_bar['price_close'] = state['trades']['price'][-1]
    new_bar['price_low'] = state['price_min']
    new_bar['price_high'] = state['price_max']
    new_bar['price_vwap'] = (np.array(state['trades']['price']) * np.array(state['trades']['volume'])).sum() / np.array(state['trades']['volume']).sum()
    new_bar['price_std'] = np.array(state['trades']['price']).std(),
    new_bar['price_range'] = state['price_range']
    new_bar['bar_return'] = state['bar_return']
    new_bar['ticks'] = state['ticks']
    new_bar['volume'] = state['volume']
    new_bar['dollars'] = state['dollar']
    new_bar['ticks_imbalance'] = state['tick_imbalance']
    new_bar['volume_imbalance'] = state['volume_imbalance']
    new_bar['dollar_imbalance'] = state['dollar_imbalance']
    return new_bar


def update_bar(tick, output_bars, state):
    
    state['trades']['epoch'].append(tick['epoch'])
    state['trades']['price'].append(tick['price'])
    state['trades']['volume'].append(tick['volume'])

    try:
        tick_side = tick_rule(first_price=price, second_price=state['trades']['price'][-2], last_side=state['trades']['side'][-1])
    except: 
        tick_side = 0
    state['trades']['side'].append(tick_side)

    # state['now_diff'] = int(datetime.datetime.utcnow().timestamp() * 1000000000) - state['trades']['epoch'][0]
    state['duration_ns'] = tick['epoch'] - state['trades']['epoch'][0]

    state['ticks'] += 1
    state['volume'] += tick['volume']
    state['dollar'] += tick['price'] * tick['volume']

    state['bar_return'] = tick['price'] - state['trades']['price'][0]
    state['price_range'], state['price_min'], state['price_max'] = update_price_stats(tick['price'], state['price_min'], state['price_max'])

    state['tick_imbalance'] += tick_side
    state['volume_imbalance'] += tick_side * tick['volume']
    state['dollar_imbalance'] += tick_side * tick['volume'] * tick['price']

    if state['thresh_duration_ns'] and state['duration_ns'] > state['thresh_duration_ns']:
        state['next_bar'] = 'duration'
    if state['thresh_ticks'] and state['ticks'] > state['thresh_ticks']:
        state['next_bar'] = 'tick_coumt'
    if state['thresh_volume'] and state['volume'] > state['thresh_volume']:
        state['next_bar'] = 'volume_sum'
    if state['thresh_dollar'] and state['dollar'] > state['thresh_dollar']:
        state['next_bar'] = 'dollar_sum'
    if state['thresh_tick_imbalance'] and state['tick_imbalance'] > state['thresh_tick_imbalance']:
        state['next_bar'] = 'tick_imbalance'
    if state['thresh_volume_imbalance'] and state['volume_imbalance'] > state['thresh_volume_imbalance']:
        state['next_bar'] = 'volumne_imbalance'        
    if state['thresh_dollar_imbalance'] and state['dollar_imbalance'] > state['thresh_dollar_imbalance']:
        state['next_bar'] = 'dollar_imbalence'
    if state['thresh_price_range'] and state['price_range'] > state['thresh_price_range']:
        state['next_bar'] = 'price_range'
    if state['thresh_return'] and abs(state['bar_return']) > state['thresh_return']:
        state['next_bar'] = 'bar_return'
    if state['thresh_renko']:
        try:
            state['thresh_renko_bull'], state['thresh_renko_bear'] = get_next_renko_thresh(
                thresh_renko=state['thresh_renko'], 
                last_return=output_bars[-1]['bar_return']
                )
        except:
            state['thresh_renko_bull'] = state['thresh_renko']
            state['thresh_renko_bear'] = -state['thresh_renko']

        if state['bar_return'] > state['thresh_renko_bull']:
            state['next_bar'] = 'renko'
        if state['bar_return'] < state['thresh_renko_bear']:
            state['next_bar'] = 'renko'

    if state['next_bar'] != 'waiting':
        print('new bar: ', state['next_bar'])
        # save new bar
        new_bar = save_bar(state)    
        output_bars.append(new_bar)
        # reset counter vars
        state = reset_state()

    return output_bars, state
