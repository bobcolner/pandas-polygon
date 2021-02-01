import datetime as dt
import numpy as np
import pandas as pd
from polygon_s3 import fetch_date_df
from polygon_ds import get_dates_df
from bar_samples import build_bars
from bar_labels import label_bars
from utils_filters import jma_filter_df


def bar_dates_stats(stacked_df: pd.DataFrame) -> pd.DataFrame:

    bars_df = stacked_df[stacked_df.bar_trigger != 'gap_filler'].reset_index(drop=True)

    bars_df.loc[:, 'date'] = bars_df.close_at.dt.date.astype('string')

    return bars_df.groupby('date').agg(
        bar_count=pd.NamedAgg(column="duration_min", aggfunc="count"),
        duration_min_median=pd.NamedAgg(column="duration_min", aggfunc="median"),
        jma_range_mean=pd.NamedAgg(column="jma_range", aggfunc="mean"),
        first_bar_open=pd.NamedAgg(column="open_at", aggfunc="min"),
        last_bar_close=pd.NamedAgg(column="close_at", aggfunc="max"),
    ).reset_index()


def get_symbol_vol_filter(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    # get exta 10 days
    adj_start_date = (dt.datetime.fromisoformat(start_date) - dt.timedelta(days=10)).date().isoformat()
    # get market daily from pyarrow dataset
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


def fill_gap(bar_1: dict, bar_2: dict, renko_size: float, fill_col: str) -> dict:

    num_steps = round(abs(bar_1[fill_col] - bar_2[fill_col]) / (renko_size / 2))
    fill_values = list(np.linspace(start=bar_1[fill_col], stop=bar_2[fill_col], num=num_steps))
    fill_values.insert(-1, bar_2[fill_col])
    fill_values.insert(-1, bar_2[fill_col])
    fill_dt = pd.date_range(
        start=bar_1['close_at'] + dt.timedelta(hours=1),
        end=bar_2['open_at'] - dt.timedelta(hours=1),
        periods=num_steps + 2,
        )
    fill_dict = {
        'bar_trigger': 'gap_filler',
        'close_at': fill_dt,
        fill_col: fill_values,
    }
    return pd.DataFrame(fill_dict).to_dict(orient='records')


def fill_gaps_dates(bar_dates: list, fill_col: str) -> pd.DataFrame:

    for idx, date in enumerate(bar_dates):
        if idx == 0:
            continue

        try:
            gap_fill = fill_gap(
                bar_1=bar_dates[idx-1]['bars'][-1],
                bar_2=bar_dates[idx]['bars'][1],
                renko_size=bar_dates[idx]['thresh']['renko_size'],
                fill_col=fill_col,
            )
            bar_dates[idx-1]['bars'] = bar_dates[idx-1]['bars'] + gap_fill
        except:
            print(date['date'])
            continue
    # build continous 'stacked' bars df
    stacked = []
    for date in bar_dates:
        stacked = stacked + date['bars']

    stacked_bars_df = pd.DataFrame(stacked)

    return stacked_bars_df


def bar_workflow(symbol: str, date: str, thresh: dict, add_label: bool=True) -> dict:
    
    # get ticks
    ticks_df = fetch_date_df(symbol, date, tick_type='trades')
    
    # filter market hours
    ticks_df.loc[:, 'nyc_dt'] = ticks_df['sip_dt'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    ticks_df = ticks_df.set_index('nyc_dt').between_time('09:30:00', '16:00:00').reset_index()
    
    # sample bars
    bars, filtered_ticks = build_bars(ticks_df, thresh)
    
    # clean ticks
    ft_ticks_df = pd.DataFrame(filtered_ticks)
    ft_ticks_df['price_clean'] = ft_ticks_df['price']
    ft_ticks_df.loc[ft_ticks_df['status'] != 'clean', 'price_clean'] = None

    if add_label:
        bars = label_bars(
            bars=bars,
            ticks_df=ft_ticks_df[ft_ticks_df.status == 'clean'],
            risk_level=thresh['renko_size'],
            horizon_mins=thresh['max_duration_sec'] / 60,
            reward_ratios=thresh['label_reward_ratios'],
            )

    bar_result = {
        'symbol': symbol,
        'date': date,
        'bars': bars,
        'thresh': thresh,
        'ticks_df': ticks_df,
        'ft_ticks_df': ft_ticks_df,
        }
    return bar_result


def bar_dates_workflow(symbol: str, start_date: str, end_date: str, thresh: dict,
    add_label: bool, ray_on: bool=False) -> list:

    daily_stats_df = get_symbol_vol_filter(symbol, start_date, end_date)
    bar_dates = []
    for row in daily_stats_df.itertuples():
        if 'range_jma_lag' in daily_stats_df.columns:
            rs = max(row.range_jma_lag / thresh['renko_range_frac'], row.vwap_jma_lag * 0.0005)
            thresh.update({'renko_size': rs})

        if ray_on:
            import ray
            bar_workflow_ray = ray.remote(bar_workflow)
            bars = bar_workflow_ray.remote(symbol, row.date, thresh, add_label)
        else:
            bars = bar_workflow(symbol, row.date, thresh, add_label)

        bar_dates.append(bars)

    if ray_on:
        bar_dates = ray.get(bar_dates)

    return bar_dates
