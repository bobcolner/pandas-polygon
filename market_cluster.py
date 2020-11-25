import numpy as np
import pandas as pd
from polygon_ds import get_dates_df


def all_dates_filer(df: pd.DataFrame) -> pd.DataFrame:
    sym_count = df.groupby('symbol').count()[['open']]
    active_days = max(df.symbol.value_counts())
    passed_sym = sym_count.loc[sym_count['open'] >= active_days].index
    df_filtered = df.loc[df.symbol.isin(passed_sym)].reset_index(drop=True)
    return df_filtered


def liquidity_filter(df: pd.DataFrame, abs_dollar_cut: float, quantile_dollar_cut: float) -> pd.DataFrame:
    sym_dollar_avg = df.groupby('symbol')[['dollar_total']].mean()
    if quantile_dollar_cut:
        min_dollar = df['dollar_total'].quantile(q=qcut)
    else:
        min_dollar = abs_dollar_cut
    passed_sym = sym_dollar_avg.loc[sym_dollar_avg['dollar_total'] > min_dollar].index
    df_filtered = df.loc[df.symbol.isin(passed_sym)].reset_index(drop=True)
    return df_filtered


def add_range(df: pd.DataFrame) -> pd.DataFrame:
    daily_range = df['high'] - df['low']
    df.loc[:, 'range'] = daily_range
    pct_daily_range = daily_range / df['close']
    df.loc[:, 'pct_daily_range'] = pct_daily_range
    return df


def volitility_filter(df: pd.DataFrame) -> pd.DataFrame:
    sym_pct_range_med = df.groupby('symbol')[['pct_daily_range']].median()
    passed_sym = sym_pct_range_med.loc[sym_pct_range_med['pct_daily_range'].between(0.005, 0.2)].index
    df_filtered = df.loc[df.symbol.isin(passed_sym)].reset_index(drop=True)
    return df_filtered


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, 'close_diff'] = df['close'].diff()
    df.loc[:, 'close_log_return'] = np.diff(np.log(df['close']))
    df.loc[:, 'vwap_diff'] = df['vwap'].diff()
    df.loc[:, 'vwap_log_return'] = np.diff(np.log(df['vwap']))
    return df


def market_cluster_workflow(tick_type: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = get_dates_df(tick_type, symbol, start_date, end_date)
    df = all_dates_filer(df)
    df = liquidity_filter(df, abs_dollar_cut=500_000)
    df = add_range(df)
    df = volitility_filter(df)
    return df
    