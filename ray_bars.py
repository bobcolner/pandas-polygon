import datetime as dt
import numpy as np
import pandas as pd
import ray
from polygon_s3 import fetch_date_df
from polygon_ds import get_dates_df
from bar_samples import build_bars
from bar_labels import label_bars, get_concurrent_stats
from utils_filters import jma_filter_df
# https://robotwealth.com/zorro-132957/


@ray.remote
def build_bars_ray(symbol: str, date: str, thresh: dict) -> dict:
    # get ticks for current date
    ticks_df = fetch_date_df(symbol, date, tick_type='trades')
    # sample bars
    bars, state = build_bars(ticks_df, thresh)
    return {'symbol': symbol, 'date': date, 'thresh': thresh, 'bars': bars}


def build_bars_dates_ray(daily_stats_df: pd.DataFrame, thresh: dict, symbol: str, range_frac: int) -> list:
    futures = []
    for row in daily_stats_df.itertuples():
        if 'range_jma_lag' in daily_stats_df.columns:
            thresh.update({'renko_size': row.range_jma_lag / range_frac})
        # if 'vwap_jma_lag' in daily_stats_df.columns:
        #     thresh.update({'min_jma_range': row.vwap_jma_lag * 0.0005})
        if 'imbalance_thresh_jma_lag' in daily_stats_df.columns:
            thresh.update({'volume_imbalance': row.imbalance_thresh_jma_lag})

        bars = build_bars_ray.remote(
            symbol=symbol, 
            date=row.date,
            thresh=thresh
        )
        futures.append(bars)

    return ray.get(futures)


def process_bar_dates(daily_vol_df: pd.DataFrame, bar_dates: list, imbalance_thresh: float) -> pd.DataFrame:
    results = []
    for date_d in bar_dates:
        imbalances = []
        durations = []
        ranges = []
        for bar in date_d['bars']:
            imbalances.append(bar['volume_imbalance'])
            durations.append(bar['duration_min'])
            ranges.append(bar['price_range'])

        results.append({
            'date': date_d['date'], 
            'bar_count': len(date_d['bars']), 
            'imbalance_thresh': pd.Series(imbalances).quantile(q=imbalance_thresh),
            'duration_min_mean': pd.Series(durations).mean(),
            'duration_min_median': pd.Series(durations).median(),
            'price_range_mean': pd.Series(ranges).mean(),
            'price_range_median': pd.Series(ranges).median(),
            'thresh': date_d['thresh']
            })

    daily_bar_stats_df = jma_filter_df(pd.DataFrame(results), 'imbalance_thresh', length=5, power=1)
    daily_bar_stats_df.loc[:, 'imbalance_thresh_jma_lag'] = daily_bar_stats_df['imbalance_thresh_jma'].shift(1)
    daily_bar_stats_df = daily_bar_stats_df.dropna()
    # join output
    daily_join_df = pd.merge(left=daily_bar_stats_df, right=daily_vol_df, left_on='date', right_on='date')
    return daily_join_df


@ray.remote
def label_bars_ray(bars: list, symbol: str, date: str, risk_level: float, horizon_mins: int, reward_ratios: list) -> list:
    ticks_df = fetch_date_df(symbol, date, 'trades')
    labeled_bars = label_bars(
        bars=bars, 
        ticks_df=ticks_df, 
        risk_level=risk_level, 
        horizon_mins=horizon_mins, 
        reward_ratios=reward_ratios
        )
    labeled_bars_df = pd.DataFrame(labeled_bars)
    labeled_bars_unq = get_concurrent_stats(labeled_bars_df)
    print(symbol, date, "label % unq.", str(labeled_bars_unq['grand_avg_unq']))
    return {
        'date': date, 
        'risk_level': risk_level, 
        'horizon_mins': horizon_mins, 
        'avg_label_uniquness': labeled_bars_unq['grand_avg_unq'],
        'labeled_bars': labeled_bars
        }


def label_bars_dates_ray(bar_dates: list, symbol: str, horizon_mins: int, reward_ratios: list) -> list:

    futures = []
    for date in bar_dates:
        result = label_bars_ray.remote(
            bars=date['bars'],
            symbol=symbol,
            date=date['date'],
            risk_level=date['thresh']['renko_size'],
            horizon_mins=horizon_mins,
            reward_ratios=reward_ratios
        )
        futures.append(result)

    return ray.get(futures)


def fill_gap(bar_1: dict, bar_2: dict, renko_size: float, price_col: str='price_wmean') -> dict:

    num_steps = round(abs(bar_1[price_col] - bar_2[price_col]) / (renko_size / 2))
    fill_prices = list(np.linspace(start=bar_1[price_col], stop=bar_2[price_col], num=num_steps))
    fill_prices.insert(-1, bar_2[price_col])
    fill_prices.insert(-1, bar_2[price_col])
    fill_dt = pd.date_range(
        start=bar_1['close_at'] + dt.timedelta(hours=1),
        end=bar_2['open_at'] - dt.timedelta(hours=1),
        periods=num_steps + 2
        )
    fill_dict = {
        'bar_trigger': 'gap_filler',
        'close_at': fill_dt,
        price_col: fill_prices
    }
    return pd.DataFrame(fill_dict).to_dict(orient='records')


def fill_gaps_dates(labeled_bar_dates: list) -> tuple:
    
    for idx, date in enumerate(labeled_bar_dates):
        if idx == 0:
            continue

        gap_fill = fill_gap(
            bar_1=labeled_bar_dates[idx-1]['labeled_bars'][-1],
            bar_2=labeled_bar_dates[idx]['labeled_bars'][1],
            renko_size=labeled_bar_dates[idx]['risk_level'],
            price_col='price_wmean'
        )
        labeled_bar_dates[idx-1]['labeled_bars'] = labeled_bar_dates[idx-1]['labeled_bars'] + gap_fill
    # build continoius 'stacked' bars df
    stacked = []
    for date in labeled_bar_dates:
        stacked = stacked + date['labeled_bars']

    stacked_bars_df = pd.DataFrame(stacked)

    return labeled_bar_dates, stacked_bars_df


def get_symbol_vol_filter(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    # get exta 10 days
    adj_start_date = (dt.datetime.fromisoformat(start_date) - dt.timedelta(days=10)).date().isoformat()
    # get market daily df
    df = get_dates_df(symbol='market', tick_type='daily', start_date=adj_start_date, end_date=end_date, source='local')
    df = df.loc[df['symbol'] == symbol].reset_index(drop=True)
    # range/volitiliry metric
    df.loc[:, 'range'] = df['high'] - df['low']
    df = jma_filter_df(df, col='range', length=5, power=1)
    df.loc[:, 'range_jma_lag'] = df['range_jma'].shift(1)
    # recent price/value metric
    df.loc[:, 'price_close_lag'] = df['close'].shift(1)
    df = jma_filter_df(df, col='vwap', length=7, power=1)
    df.loc[:, 'vwap_jma_lag'] = df['vwap_jma'].shift(1)
    return df.dropna().reset_index(drop=True)


def bars_workflow_ray(symbol: str, start_date: str, end_date: str, thresh: dict) -> tuple:
    # calculate daily ATR filter
    daily_vol_df = get_symbol_vol_filter(symbol, start_date, end_date)
    # 1st pass bar sampeing based on ATR
    bar_dates = build_bars_dates_ray(
        daily_stats_df=daily_vol_df,
        thresh=thresh,
        symbol=symbol,
        range_frac=12
        )
    # calcuate stats on 1st pass bar samples
    daily_bar_stats_df = process_bar_dates(daily_vol_df, bar_dates, 0.95)
    # 2ed pass bar sampleing based on ATR and imbalance threshold
    bar_dates = build_bars_dates_ray(
        daily_stats_df=daily_bar_stats_df,
        thresh=thresh,
        symbol=symbol,
        range_frac=15
        )
    # calcuate stats on 2ed pass bar samples
    daily_bar_stats_df = process_bar_dates(daily_vol_df, bar_dates, 0.95)
    # label 2ed pass bar samples
    labeled_bar_dates = label_bars_dates_ray(
        bar_dates, 
        symbol, 
        label_length_mins=30, 
        label_reward_ratios=list(np.arange(3, 12, 1))
        )
    # fill daily gaps
    labeled_bar_dates, stacked_bars_df = fill_gaps_dates(labeled_bar_dates)

    return daily_bar_stats_df, labeled_bar_dates, stacked_bars_df
