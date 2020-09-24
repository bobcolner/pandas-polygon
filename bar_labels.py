import pandas as pd
from tqdm import tqdm
import ipdb # ipdb.set_trace(context=10)


def triple_barrier_outcomes(ticks_df:pd.DataFrame, label_start_at:pd._libs.tslibs.timestamps.Timestamp, 
    risk_level:float, label_horizon_mins:int, reward_base:float=1.0, debug=False):
    # get label price series
    price_start_at = label_start_at + pd.Timedelta(value=3, unit='seconds') # inference+network latency compensation
    price_end_at = label_start_at + pd.Timedelta(value=label_horizon_mins, unit='minutes')
    label_prices = ticks_df[(ticks_df['epoch'] >= price_start_at.value) & (ticks_df['epoch'] < price_end_at.value)]
    # set profit and stop loss levels (long side)
    long_stop_price = label_prices['price'].values[0] - risk_level
    long_profit_price2 = label_prices['price'].values[0] + (risk_level * reward_base * 2)
    long_profit_price3 = label_prices['price'].values[0] + (risk_level * reward_base * 3)
    long_profit_price4 = label_prices['price'].values[0] + (risk_level * reward_base * 4)
    long_profit_price5 = label_prices['price'].values[0] + (risk_level * reward_base * 5)
    # short side
    short_stop_price = label_prices['price'].values[0] + risk_level
    short_profit_price2 = label_prices['price'].values[0] - (risk_level * reward_base * 2)
    short_profit_price3 = label_prices['price'].values[0] - (risk_level * reward_base * 3)
    short_profit_price4 = label_prices['price'].values[0] - (risk_level * reward_base * 4)
    short_profit_price5 = label_prices['price'].values[0] - (risk_level * reward_base * 5)
    # determine what targets are hit
    long_stop_hit_at = label_prices[label_prices['price'] <= long_stop_price].min().date_time
    long_profit2_hit_at = label_prices[label_prices['price'] >= long_profit_price2].min().date_time
    long_profit3_hit_at = label_prices[label_prices['price'] >= long_profit_price3].min().date_time
    long_profit4_hit_at = label_prices[label_prices['price'] >= long_profit_price4].min().date_time
    long_profit5_hit_at = label_prices[label_prices['price'] >= long_profit_price5].min().date_time
    # short side target git
    short_stop_hit_at = label_prices[label_prices['price'] >= short_stop_price].min().date_time
    short_profit2_hit_at = label_prices[label_prices['price'] <= short_profit_price2].min().date_time
    short_profit3_hit_at = label_prices[label_prices['price'] <= short_profit_price3].min().date_time
    short_profit4_hit_at = label_prices[label_prices['price'] <= short_profit_price4].min().date_time
    short_profit5_hit_at = label_prices[label_prices['price'] <= short_profit_price5].min().date_time
    # find first (if any) target outcome
    outcomes = pd.DataFrame({
        'label_rrr': [0, reward_base * 2, reward_base *  3, reward_base * 4, reward_base * 5, 0, reward_base * -2, reward_base * -3, reward_base * -4, reward_base * -5],
        'label_side': ['long', 'long', 'long', 'long', 'long', 'short', 'short', 'short', 'short', 'short'],
        'label_outcome': ['stop', 'profit', 'profit', 'profit', 'profit', 'stop', 'profit', 'profit', 'profit', 'profit'],
        'label_outcome_at': [long_stop_hit_at, long_profit2_hit_at, long_profit3_hit_at, long_profit4_hit_at, long_profit5_hit_at, short_stop_hit_at, short_profit2_hit_at, short_profit3_hit_at, short_profit4_hit_at, short_profit5_hit_at],
        'target_price': [long_stop_price, long_profit_price2, long_profit_price3, long_profit_price4, long_profit_price5, short_stop_price, short_profit_price2, short_profit_price3, short_profit_price4, short_profit_price5],
        'outcome_price': [long_stop_price, long_profit_price2, long_profit_price3, long_profit_price4, long_profit_price5, short_stop_price, short_profit_price2, short_profit_price3, short_profit_price4, short_profit_price5],
    }).sort_values('label_outcome_at')
    outcomes['price_diff'] = outcomes['outcome_price'] - label_prices['price'].values[0]
    outcomes['pct_change'] = (outcomes['price_diff'] / label_prices['price'].values[0]) * 100
    if debug:
        return outcomes, label_prices
    else:
        return outcomes


def signed_outcomes_to_label(outcomes):
    outcomes = outcomes.dropna()
    if outcomes.shape[0] == 0: # no outcomes
        # print('neutral')
        label = [{
                    'label_rrr': 0,
                    'label_outcome': 'neutral',
                    'label_side': 'both',
                }]
    elif outcomes[outcomes['label_outcome']=='stop'].shape[0] > 0: # stop-loss found
        idx = outcomes[outcomes['label_outcome']=='stop'].index.values[0] # index of stop
        if idx == 0: # stop-loss is first outcome
            # print('stop')
            label = outcomes.head(1).to_dict(orient='records')
        elif idx > 0:
            # print('profit before stop')
            multi_label = outcomes[outcomes.index < idx] # profit outcomes before stop-out
            label = multi_label[multi_label['label_rrr'].abs()==multi_label['label_rrr'].abs().max()].to_dict(orient='records')
    else:
        # print('profit')
        label = outcomes[outcomes['label_rrr'].abs()==outcomes['label_rrr'].abs().max()].to_dict(orient='records') # get max reward
    return label[0]


def outcomes_to_label(outcomes, price_end_at):
    long_outcomes = outcomes.loc[outcomes['label_side']=='long'].reset_index(drop=True)
    short_outcomes = outcomes.loc[outcomes['label_side']=='short'].reset_index(drop=True)
    long_label = signed_outcomes_to_label(long_outcomes)
    short_label = signed_outcomes_to_label(short_outcomes)
    if (long_label['label_outcome'] == 'profit') and (short_label['label_outcome'] == 'stop'):
        label = long_label
    elif (short_label['label_outcome'] == 'profit') and (long_label['label_outcome'] == 'stop'):
        label = short_label
    elif (short_label['label_outcome'] in ['neutral', 'stop']) and (long_label['label_outcome'] in ['neutral', 'stop']):
        try:
            price_end_at = max(long_label['label_outcome_at'], short_label['label_outcome_at'])
        except: 
            price_end_at = price_end_at
        label = {'label_outcome': 'neutral', 'label_side': 'both', 'label_rrr': 0, 'label_outcome_at': price_end_at}
    else:
        label = {'label_outcome': 'unknown'}
    return label


def label_bars_triple_barrier(bars:list, ticks_df:pd.DataFrame, risk_level:float, 
    label_horizon_mins:int, reward_base:float=1.0) -> list:
    for idx, row in tqdm(enumerate(bars), total=len(bars)):
        label_start_at = row['close_at']
        outcomes = triple_barrier_outcomes(ticks_df, label_start_at, risk_level, label_horizon_mins, reward_base)
        label = outcomes_to_label(outcomes, label_start_at)
        bars[idx].update(label)
    return bars

