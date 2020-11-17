import numpy as np
import pandas as pd


def all_dates_filer(df: pd.DataFrame) -> pd.DataFrame:
    sym_count = df.groupby('symbol').count()['open']
    # print(df.symbol.value_counts().describe())
    active_days = max(df.symbol.value_counts())
    passed_sym = sym_count.loc[sym_count['open'] >= active_days].index
    df_filtered = df.loc[df.symbol.isin(passed_sym)].reset_index(drop=True)
    return df_filtered


def liquidity_filter(df: pd.DataFrame) -> pd.DataFrame:
    sym_dolavg = df.groupby('symbol')[['dollar_total']].mean()
    # print(df['dollar_total'].describe())
    min_dolavg = df['dollar_total'].quantile(q=0.25)
    passed_sym = sym_dolavg.loc[sym_dolavg['dollar_total'] > min_dolavg].index
    df_filtered = df.loc[df.symbol.isin(passed_sym)].reset_index(drop=True)
    return df_filtered


def volitility_filter(df: pd.DataFrame) -> pd.DataFrame:
    daily_range = df['high'] - df['low']
    df.loc[:, 'range'] = daily_range
    pct_daily_range = daily_range / df['close']
    df.loc[:, 'pct_daily_range'] = pct_daily_range
    sym_pct_range_avg = df.groupby('symbol')[['pct_daily_range']].mean()
    passed_sym = sym_pct_range_avg.loc[sym_pct_range_avg['pct_daily_range'] < 0.5].index
    df_filtered = df.loc[df.symbol.isin(passed_sym)].reset_index(drop=True)
    return df


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

