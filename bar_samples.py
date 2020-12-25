import pandas as pd


def time_bars(ticks_df: pd.DataFrame, date: str, freq: str='15min') -> list:
    import datetime as dt
    from tqdm import tqdm

    start_date = dt.datetime.strptime(date, '%Y-%m-%d')
    end_date = start_date + dt.timedelta(days=1)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq, tz='utc', closed=None)
    bars = []
    for i in tqdm(list(range(len(dates)-2))):
        ticks = ticks_df.loc[(ticks_df['date_time'] >= dates[i]) & (ticks_df['date_time'] < dates[i+1])]
        _, state = build_bars(ticks_df=ticks, thresh={})
        bar = output_new_bar(state)
        bar['open_at'] = dates[i]
        bar['close_at'] = dates[i+1]
        bars.append(bar)

    return bars


def tick_rule(latest_price: float, prev_price: float, last_side: int=0) -> int:
    try:
        diff = latest_price - prev_price
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


def imbalance_runs(state: dict) -> dict:
    if len(state['trades']['side']) >= 2:
        if state['trades']['side'][-1] == state['trades']['side'][-2]:
            state['stat']['tick_run'] += 1        
            state['stat']['volume_run'] += state['trades']['volume'][-1]
            state['stat']['dollar_run'] += state['trades']['price'][-1] * state['trades']['volume'][-1]
        else:
            state['stat']['tick_run'] = 0
            state['stat']['volume_run'] = 0
            state['stat']['dollar_run'] = 0
    # state['stat']['tick_run_max'] = state['stat']['tick_run'] if state['stat']['tick_run'] > state['stat']['tick_run_max'] else state['stat']['tick_run_max']
    # state['stat']['volume_run_max'] = state['stat']['volume_run'] if state['stat']['volume_run'] > state['stat']['volume_run_max'] else state['stat']['volume_run_max']
    # state['stat']['dollar_run_max'] = state['stat']['dollar_run'] if state['stat']['dollar_run'] > state['stat']['dollar_run_max'] else state['stat']['dollar_run_max']
    return state


def imbalance_net(state: dict) -> dict:
    state['stat']['tick_imbalance'] += state['trades']['side'][-1]
    state['stat']['volume_imbalance'] += state['trades']['side'][-1] * state['trades']['volume'][-1]
    state['stat']['dollar_imbalance'] += state['trades']['side'][-1] * state['trades']['volume'][-1] * state['trades']['price'][-1]
    return state


def reset_state(thresh: dict={}) -> dict:
    state = {}    
    state['thresh'] = thresh
    state['stat'] = {}
    # accumulators
    state['stat']['duration_sec'] = 0
    state['stat']['price_min'] = 10 ** 5
    state['stat']['price_max'] = 0
    state['stat']['price_range'] = 0
    state['stat']['price_return'] = 0
    state['stat']['jma_min'] = 10 ** 5
    state['stat']['jma_max'] = 0
    state['stat']['jma_range'] = 0
    state['stat']['jma_return'] = 0
    state['stat']['tick_count'] = 0
    state['stat']['volume'] = 0
    state['stat']['dollars'] = 0
    state['stat']['tick_imbalance'] = 0
    state['stat']['volume_imbalance'] = 0
    state['stat']['dollar_imbalance'] = 0
    # state['stat']['tick_run'] = 0
    # state['stat']['volume_run'] = 0
    # state['stat']['dollar_run'] = 0
    # copy of tick events
    state['trades'] = {}
    state['trades']['date_time'] = []
    state['trades']['price'] = []
    state['trades']['volume'] = []
    state['trades']['side'] = []
    state['trades']['jma'] = []
    # trigger status
    state['trigger_yet?!'] = 'waiting'
    return state


def output_new_bar(state: dict) -> dict:
    from statsmodels.stats.weightstats import DescrStatsW
    
    new_bar = {}
    if state['stat']['tick_count'] == 0:
        return new_bar

    new_bar['bar_trigger'] = state['trigger_yet?!']
    # time
    new_bar['open_at'] = state['trades']['date_time'][0]
    new_bar['close_at'] = state['trades']['date_time'][-1]
    new_bar['duration_td'] = new_bar['close_at'] - new_bar['open_at']    
    new_bar['duration_sec'] = state['stat']['duration_sec']
    new_bar['duration_min'] = new_bar['duration_sec'] / 60
    # price
    new_bar['price_open'] = state['trades']['price'][0]
    new_bar['price_close'] = state['trades']['price'][-1]
    new_bar['price_low'] = state['stat']['price_min']
    new_bar['price_high'] = state['stat']['price_max']
    new_bar['price_range'] = state['stat']['price_range']
    new_bar['price_return'] = state['stat']['price_return']
    # volume weighted price
    dsw = DescrStatsW(data=state['trades']['price'], weights=state['trades']['volume'])
    qtiles = dsw.quantile(probs=[0.1, 0.5, 0.9]).values
    new_bar['price_wq10'] = qtiles[0]
    new_bar['price_wq50'] = qtiles[1]
    new_bar['price_wq90'] = qtiles[2]
    new_bar['price_wq_range'] = new_bar['price_wq90'] - new_bar['price_wq10']
    new_bar['price_wmean'] = dsw.mean
    new_bar['price_wstd'] = dsw.std
    # jma
    new_bar['jma_open'] = state['trades']['jma'][0]
    new_bar['jma_close'] = state['trades']['jma'][-1]
    new_bar['jma_low'] = state['stat']['jma_min']
    new_bar['jma_high'] = state['stat']['jma_max']
    new_bar['jma_range'] = state['stat']['jma_range']
    new_bar['jma_return'] = state['stat']['jma_return']
    # volume weighted jma
    dsw = DescrStatsW(data=state['trades']['jma'], weights=state['trades']['volume'])
    qtiles = dsw.quantile(probs=[0.1, 0.5, 0.9]).values
    new_bar['jma_wq10'] = qtiles[0]
    new_bar['jma_wq50'] = qtiles[1]
    new_bar['jma_wq90'] = qtiles[2]
    new_bar['jma_wq_range'] = new_bar['jma_wq90'] - new_bar['jma_wq10']
    new_bar['jma_wmean'] = dsw.mean
    new_bar['jma_wstd'] = dsw.std
    # tick/vol/dollar/imbalance
    new_bar['tick_count'] = state['stat']['tick_count']
    new_bar['volume'] = state['stat']['volume']
    new_bar['dollars'] = state['stat']['dollars']
    new_bar['tick_imbalance'] = state['stat']['tick_imbalance']
    new_bar['volume_imbalance'] = state['stat']['volume_imbalance']
    new_bar['dollar_imbalance'] = state['stat']['dollar_imbalance']
    # new_bar['tick_imbalance_run'] = state['stat']['tick_run']
    # new_bar['volume_imbalance_run'] = state['stat']['volume_run']
    # new_bar['dollar_imbalance_run'] = state['stat']['dollar_run']
    return new_bar


def check_bar_thresholds(state: dict) -> dict:

    def get_next_renko_thresh(renko_size: float, last_bar_return: float, reversal_multiple: float) -> tuple:
        if last_bar_return >= 0:
            thresh_renko_bull = renko_size
            thresh_renko_bear = -renko_size * reversal_multiple
        elif last_bar_return < 0:
            thresh_renko_bull = renko_size * reversal_multiple
            thresh_renko_bear = -renko_size
        return thresh_renko_bull, thresh_renko_bear

    if 'renko_size' in state['thresh']:
        try:
            state['thresh']['renko_bull'], state['thresh']['renko_bear'] = get_next_renko_thresh(
                renko_size=state['thresh']['renko_size'],
                last_bar_return=state['stat']['last_bar_return'],
                reversal_multiple=state['thresh']['renko_reveral_multiple']
            )
        except:
            state['thresh']['renko_bull'] = state['thresh']['renko_size']
            state['thresh']['renko_bear'] = -state['thresh']['renko_size']

        if state['stat'][state['thresh']['renko_return']] >= state['thresh']['renko_bull']:
            state['trigger_yet?!'] = 'renko_up'
        if state['stat'][state['thresh']['renko_return']] < state['thresh']['renko_bear']:
            state['trigger_yet?!'] = 'renko_down'

    if 'max_duration_sec' in state['thresh'] and state['stat']['duration_sec'] > state['thresh']['max_duration_sec']:
        state['trigger_yet?!'] = 'duration'

    if 'volume_imbalance' in state['thresh'] and abs(state['stat']['volume_imbalance']) >= state['thresh']['volume_imbalance']:
        state['trigger_yet?!'] = 'volume_imbalance'

    # override newbar trigger with 'minimum' thresholds
    if 'min_duration_sec' in state['thresh'] and state['stat']['duration_sec'] < state['thresh']['min_duration_sec']:
        state['trigger_yet?!'] = 'waiting'

    if 'min_tick_count' in state['thresh'] and state['stat']['tick_count'] < state['thresh']['min_tick_count']:
        state['trigger_yet?!'] = 'waiting'

    if 'min_price_range' in state['thresh'] and state['stat']['price_range'] < state['thresh']['min_price_range']:
        state['trigger_yet?!'] = 'waiting'

    if 'min_jma_range' in state['thresh'] and state['stat']['jma_range'] < state['thresh']['min_jma_range']:
        state['trigger_yet?!'] = 'waiting'

    return state


def update_bar_state(tick: dict, state: dict, bars: list, thresh: dict={}) -> tuple:

    state['trades']['date_time'].append(tick['date_time'])
    state['trades']['price'].append(tick['price'])
    state['trades']['jma'].append(tick['jma'])
    state['trades']['volume'].append(tick['volume'])

    if len(state['trades']['price']) >= 2:
        tick_side = tick_rule(
            latest_price=state['trades']['price'][-1],
            prev_price=state['trades']['price'][-2],
            last_side=state['trades']['side'][-1]
            )
    else:
        tick_side = 0
    state['trades']['side'].append(tick_side)

    state = imbalance_net(state)
    # state = imbalance_runs(state)
    state['stat']['duration_sec'] = (tick['date_time'].value - state['trades']['date_time'][0].value) // 10**9
    state['stat']['tick_count'] += 1
    state['stat']['volume'] += tick['volume']
    state['stat']['dollars'] += tick['price'] * tick['volume']
    # price
    state['stat']['price_min'] = tick['price'] if tick['price'] < state['stat']['price_min'] else state['stat']['price_min']
    state['stat']['price_max'] = tick['price'] if tick['price'] > state['stat']['price_max'] else state['stat']['price_max']
    state['stat']['price_range'] = state['stat']['price_max'] - state['stat']['price_min']
    state['stat']['price_return'] = tick['price'] - state['trades']['price'][0]
    state['stat']['last_bar_return'] = bars[-1]['price_return'] if len(bars) > 0 else 0
    # jma
    state['stat']['jma_min'] = tick['jma'] if tick['jma'] < state['stat']['jma_min'] else state['stat']['jma_min']
    state['stat']['jma_max'] = tick['jma'] if tick['jma'] > state['stat']['jma_max'] else state['stat']['jma_max']
    state['stat']['jma_range'] = state['stat']['jma_max'] - state['stat']['jma_min']
    state['stat']['jma_return'] = tick['jma'] - state['trades']['jma'][0]
    # check state tirggered sample threshold
    state = check_bar_thresholds(state)
    if state['trigger_yet?!'] != 'waiting':
        new_bar = output_new_bar(state)
        bars.append(new_bar)
        state = reset_state(thresh)
    
    return bars, state


def filter_tick(tick: dict, state: list, jma_length: int=7, jma_power: float=2.0) -> tuple:
    from utils_filters import jma_filter_update

    jma_state = jma_filter_update(
        value=tick['price'],
        state=state[-1]['jma_state'],
        length=jma_length,
        power=jma_power,
        phase=0.0,
        )
    tick.update({ # add jma features to 'tick'
        'jma': jma_state['jma'],
        'pct_diff': (tick['price'] - jma_state['jma']) / jma_state['jma'],
        'jma_state': jma_state,
        })
    state.append(tick) # add new tick to buffer
    state = state[-100:] # keep most recent items

    tick['ts_diff'] = abs(tick['sip_dt'] - tick['exchange_dt'])

    if len(state) <= (jma_length + 1):  # filling window/buffer
        tick['status'] = 'filter_warm_up'
    elif tick['date_time'] < '8am nyc' and tick['date_time'] > '6pm nyc':
        tick['status'] = 'after_hours'
    elif tick['volume'] < 1:  # zero volume/size tick
        tick['status'] = 'zero_volume'
    elif tick['irregular'] == True:  # 'irrgular' tick condition
        tick['status'] = 'irregular_tick_condition'
    elif abs(tick['sip_dt'] - tick['exchange_dt']) > pd.to_timedelta(2, unit='S'): # remove large ts deltas
        tick['status'] = 'timestamp_diff'
    elif abs(tick['pct_diff']) > 0.001:  # jma filter outlier
        tick['status'] = 'filter_outlier'
    else:
        tick['status'] = 'clean'

    if (tick['status'] not in ['clean', 'filter_warm_up']):
        state.pop(-1)
    # format tick
    tick['date_time'] = tick['sip_dt']
    tick.pop('sip_dt', None)
    tick.pop('exchange_dt', None)
    tick.pop('irregular', None)
    return tick, state


def build_bars(ticks_df: pd.DataFrame, thresh: dict) -> tuple:
    
    filter_state = [{'jma_state': {
        'e0': ticks_df.price.values[0],
        'e1': 0.0, 'e2': 0.0,
        'jma': ticks_df.price.values[0],
        }}]
    bar_state = reset_state(thresh)
    bars = []
    ticks = []
    for t in ticks_df.itertuples():
        tick = {
            'sip_dt': t.sip_dt,
            'exchange_dt': t.exchange_dt,
            'price': t.price,
            'volume': t.size,
            'irregular': t.irregular,
            'status': 'new',
        }
        tick, filter_state = filter_tick(tick, filter_state)
        if tick['status'] == 'clean':
            bars, bar_state = update_bar_state(tick, bar_state, bars, thresh)

        ticks.append(tick)

    return bars, ticks
