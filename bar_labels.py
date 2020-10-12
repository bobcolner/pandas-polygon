import statsmodels.api as sm
import pandas as pd
from tqdm import tqdm


def get_tb_outcome(reward_ratio:float, risk_level:float, side:str, label_prices:pd.DataFrame, goal='profit'):
    first_price = label_prices['price'].values[0]
    if side=='long':
        if goal=='profit':
            target_price = first_price + (risk_level * reward_ratio)
            target_at = label_prices[label_prices['price'] >= target_price].min().date_time
        elif goal=='stop':
            target_price = first_price - risk_level
            target_at = label_prices[label_prices['price'] < target_price].min().date_time
            reward_ratio = -1
    elif side=='short':
        if goal=='profit':
            target_price = first_price - (risk_level * reward_ratio)
            target_at = label_prices[label_prices['price'] <= target_price].min().date_time
        elif goal=='stop':
            target_price = first_price + risk_level
            target_at = label_prices[label_prices['price'] > target_price].min().date_time
            reward_ratio = -1
        reward_ratio = reward_ratio * -1
    outcome = {
        'label_rrr': reward_ratio,
        'label_side': side,
        'label_outcome': goal,
        'label_outcome_at': target_at,
        'label_price_pct_change': (first_price - target_price) / (first_price * 100)
    }
    return outcome


def triple_barrier_outcomes(label_prices:pd.DataFrame, risk_level:float, reward_ratios:list) -> list:
    first_price = label_prices['price'].values[0]    
    tb_outcomes = []
    for side in ['long', 'short']:
        stop_outcome = get_tb_outcome(None, risk_level, side, label_prices, goal='stop')
        tb_outcomes.append(stop_outcome)
        for reward in reward_ratios:
            profit_outcome = get_tb_outcome(reward, risk_level, side, label_prices, goal='profit')
            tb_outcomes.append(profit_outcome)
    tb_df = pd.DataFrame(tb_outcomes).sort_values('label_outcome_at')
    return tb_df


def signed_outcomes_to_label(outcomes:pd.DataFrame) -> dict:
    outcomes = outcomes.dropna()
    if outcomes.shape[0] == 0: # no outcomes
        # print('neutral')
        label = [{
                    'label_rrr': 0,
                    'label_outcome': 'neutral',
                    'label_side': 'neutral',
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


def outcomes_to_label(outcomes:pd.DataFrame, price_end_at:pd._libs.tslibs.timestamps.Timestamp) -> dict:
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
        label = {'label_outcome': 'neutral', 'label_side': 'neutral', 'label_rrr': 0, 'label_outcome_at': price_end_at}
    else:
        label = {'label_outcome': 'unknown'}
    return label


def get_trend_outcome(label_prices):
    if len(label_prices) < 10:
        return {}
    df = label_prices.copy()
    df['const'] = 1
    df = df.reset_index()
    model = sm.OLS(endog=df['price'], exog=df[['const', 'index']])
    results = model.fit()
    trend = {
        'label_trend_slope': results.params[1],
        'label_trend_tvalue': results.tvalues[0],
        'label_trend_r2': results.rsquared,
    }
    return trend


def get_label_ticks(ticks_df:pd.DataFrame, label_start_at:pd._libs.tslibs.timestamps.Timestamp, label_horizon_mins:int) -> pd.DataFrame:
    price_start_at = label_start_at + pd.Timedelta(value=3, unit='seconds') # inference+network latency compensation
    price_end_at = label_start_at + pd.Timedelta(value=label_horizon_mins, unit='minutes')
    label_prices = ticks_df.loc[(ticks_df['epoch'] >= price_start_at.value) & (ticks_df['epoch'] < price_end_at.value)]
    return label_prices,  price_end_at


def label_bars(bars:list, ticks_df:pd.DataFrame, risk_level:float, label_horizon_mins:int, reward_ratios:list) -> list:

    for idx, row in tqdm(enumerate(bars), total=len(bars)):
        label_prices, price_end_at = get_label_ticks(ticks_df, row['close_at'], label_horizon_mins)
        outcomes = triple_barrier_outcomes(label_prices, risk_level, reward_ratios)
        label = outcomes_to_label(outcomes, price_end_at)
        bars[idx].update(label)
        trend = get_trend_outcome(label_prices)
        bars[idx].update(trend)

    return bars
