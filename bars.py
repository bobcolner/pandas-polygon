import os
import datetime
import numpy as np
import pandas as pd


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


def reset_state(thresh={}):
    state = {}    
    state['thresh'] = thresh
    # accululators
    state['duration_ns'] = 0
    state['duration_sec'] = 0
    state['duration_min'] = 0
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
    state['bar_trigger_state'] = 'waiting'
    return state


def save_bar(state):
    new_bar = {}
    if state['tick_count'] == 0:
        return new_bar
    new_bar['bar_trigger'] = state['bar_trigger_state']
    new_bar['open_at'] = pd.to_datetime(state['trades']['epoch'][0], utc=True, unit='ns')
    new_bar['close_at'] = pd.to_datetime(state['trades']['epoch'][-1], utc=True, unit='ns')
    new_bar['duration_sec'] = (new_bar['close_at'] - new_bar['open_at']).total_seconds()
    new_bar['duration_min'] = new_bar['duration_sec'] / 60
    new_bar['duration_ns'] = state['duration_ns']
    # new_bar['duration_dt'] = datetime.duration()? todo
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
    return new_bar


def get_next_renko_thresh(thresh_renko, last_return):
    if last_return >= 0:
        thresh_renko_bull = thresh_renko
        thresh_renko_bear = -thresh_renko * 2
    elif last_return < 0:
        thresh_renko_bull = thresh_renko * 2
        thresh_renko_bear = -thresh_renko
    return thresh_renko_bull, thresh_renko_bear


def bar_thresh(state, last_bar_return=0):
    if 'duration_ns' in state['thresh'] and state['duration_ns'] > state['thresh']['duration_ns']:
        state['bar_trigger_state'] = 'duration'
    if 'ticks' in state['thresh'] and state['tick_count'] >= state['thresh']['ticks']:
        state['bar_trigger_state'] = 'tick_coumt'
    if 'volume' in state['thresh'] and state['volume_sum'] >= state['thresh']['volume']:
        state['bar_trigger_state'] = 'volume_sum'
    if 'dollar' in state['thresh'] and state['dollar_sum'] >= state['thresh']['dollar']:
        state['bar_trigger_state'] = 'dollar_sum'
    if 'tick_imbalance' in state['thresh'] and abs(state['tick_imbalance']) >= state['thresh']['tick_imbalance']:
        state['bar_trigger_state'] = 'tick_imbalance'
    if 'volume_imbalance' in state['thresh'] and abs(state['volume_imbalance']) >= state['thresh']['volume_imbalance']:
        state['bar_trigger_state'] = 'volume_imbalance'        
    if 'dollar_imbalance' in state['thresh'] and abs(state['dollar_imbalance']) >= state['thresh']['dollar_imbalance']:
        state['bar_trigger_state'] = 'dollar_imbalence'
    if 'price_range' in state['thresh'] and state['price_range'] >= state['thresh']['price_range']:
        state['bar_trigger_state'] = 'price_range'
    if 'return' in state['thresh'] and abs(state['bar_return']) >= state['thresh']['return']:
        state['bar_trigger_state'] = 'bar_return'
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
            state['bar_trigger_state'] = 'renko'
        if state['bar_return'] < state['thresh']['renko_bear']:
            state['bar_trigger_state'] = 'renko'
    if 'tick_run' in state['thresh'] and state['tick_run'] >= state['thresh']['tick_run']:
        state['bar_trigger_state'] = 'tick_run'
    if 'volume_run' in state['thresh'] and state['volume_run'] >= state['thresh']['volume_run']:
        state['bar_trigger_state'] = 'volume_run'
    if 'dollar_run' in state['thresh'] and state['dollar_run'] >= state['thresh']['dollar_run']:
        state['bar_trigger_state'] = 'dollar_run'

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


def update_bar(tick, output_bars, state, thresh={}, outliner_pct_change = 0.002):
    
    if tick['volume'] <= 0:
        print('dropping zero volume tick')
        print(tick)
        return output_bars, state

    state['trades']['epoch'].append(tick['epoch'])
    state['trades']['price'].append(tick['price'])
    state['trades']['volume'].append(tick['volume']) 

    if len(state['trades']['price']) >= 2:
        
        tick_side = tick_rule(
            first_price=state['trades']['price'][-1], 
            second_price=state['trades']['price'][-2], 
            last_side=state['trades']['side'][-1]
            )
    else: 
        tick_side = 0

    state['trades']['side'].append(tick_side)
    state = imbalance_net(state)
    state = imbalance_runs(state)
    # state['tick_latency'] = int(datetime.datetime.utcnow().timestamp() * 1000000000) - state['trades']['epoch'][0]
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

    if state['bar_trigger_state'] != 'waiting':
        new_bar = save_bar(state)
        output_bars.append(new_bar)
        # reset counter vars
        state = reset_state(thresh)

    return output_bars, state


def build_bars(df, thresh={}, as_df=True):
    state = reset_state(thresh)
    output_bars = []
    nrow = 0
    while nrow < len(df):
        tick = {
            'epoch': df['epoch'].values[nrow],
            'price': df['price'].values[nrow],
            'volume': df['volume'].values[nrow]
        }
        output_bars, state = update_bar(tick, output_bars, state, thresh)
        nrow += 1

    if as_df:
        output_bars = pd.DataFrame(output_bars).set_index('close_at', drop=True)

    return output_bars, state


def time_bars(df, date, freq='15min', as_df=True):
    df['date_time'] = pd.to_datetime(df['epoch'], utc=True, unit='ns')
    start_date = datetime.datetime.strptime(date, '%Y-%m-%d')
    end_date = start_date + datetime.timedelta(days=1)
    dr = pd.date_range(start=start_date, end=end_date, freq=freq, tz='utc', closed=None)
    output_bars = []
    for i in list(range(len(dr)-2)):
        ticks = df[(df.date_time >= dr[i]) & (df.date_time < dr[i+1])]
        _, state = build_bars(df=ticks, thresh={})
        bar = save_bar(state)
        bar['open_at'] = dr[i]
        bar['close_at'] = dr[i+1]
        output_bars.append(bar)
    if as_df:
        output_bars = pd.DataFrame(output_bars).set_index('close_at', drop=True)
    return output_bars
