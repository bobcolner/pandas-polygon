import datetime as dt
import pandas as pd
from polygon_s3 import fetch_date_df
from polygon_ds import get_dates_df
from bar_samples import build_bars
from bar_labels import label_bars
from utils_filters import jma_filter_df


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


def bar_workflow(symbol: str, date: str, thresh: dict, add_label: bool=True) -> tuple:
    # get ticks
    ticks_df = fetch_date_df(symbol, date, tick_type='trades')
    ticks_df = ticks_df.set_index('sip_dt').between_time('14:30:00', '21:00:00').reset_index()
    # sample bars
    bars, filtered_ticks = build_bars(ticks_df, thresh)
    # tick
    ticks2_df = pd.DataFrame(filtered_ticks)
    ticks2_df['price_clean'] = ticks2_df['price']
    ticks2_df.loc[ticks2_df.status != 'clean', 'price_clean'] = None
    if add_label:
        labeled_bars = label_bars(
            bars=bars,
            ticks_df=ticks2_df[ticks2_df.status == 'clean'],
            risk_level=thresh['risk_level'],
            horizon_mins=thresh['horizon_mins'],
            reward_ratios=thresh['reward_ratios'],
            )
        bars_df = pd.DataFrame(labeled_bars)
    else:
        bars_df = pd.DataFrame(bars)

    # fill daily gaps
    # labeled_bar_dates, stacked_bars_df = fill_gaps_dates(labeled_bars)

    results = {
        'symbol': symbol,
        'date': date,
        'thresh': thresh,
        'bars': bars_df,
        'ticks': ticks2_df,
        }
    return results
