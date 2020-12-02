import numpy as np
import pandas as pd
from polygon_ds import get_dates_df
from polygon_df import get_symbol_details_df


def all_dates_filer(df: pd.DataFrame) -> pd.DataFrame:
    sym_count = df.groupby('symbol')[['open']].count()
    active_days = max(df.symbol.value_counts())
    passed_sym = sym_count.loc[sym_count['open'] >= active_days].index
    df_filtered = df.loc[df.symbol.isin(passed_sym)].reset_index(drop=True)
    return df_filtered


def liquidity_filter(df: pd.DataFrame, abs_dollar_cut: float, quantile_dollar_cut: float=None) -> pd.DataFrame:
    sym_dollar_avg = df.groupby('symbol')[['dollar_total']].mean()
    if quantile_dollar_cut:
        min_dollar = df['dollar_total'].quantile(q=qcut)
    else:
        min_dollar = abs_dollar_cut
    passed_sym = sym_dollar_avg.loc[sym_dollar_avg['dollar_total'] > min_dollar].index
    df_filtered = df.loc[df.symbol.isin(passed_sym)]
    return df_filtered.reset_index(drop=True)


def add_range(df: pd.DataFrame) -> pd.DataFrame:
    daily_range = df['high'] - df['low']
    df.loc[:, 'range'] = daily_range
    range_value_pct = daily_range / df['vwap']
    df.loc[:, 'range_value_pct'] = range_value_pct
    return df


def range_value_filter(df: pd.DataFrame, low_cut: float, high_cut: float) -> pd.DataFrame:
    sym_pct_range_med = df.groupby('symbol')[['range_value_pct']].median()
    passed_sym = sym_pct_range_med.loc[sym_pct_range_med['range_value_pct'].between(low_cut, high_cut)].index
    df_filtered = df.loc[df.symbol.isin(passed_sym)]
    return df_filtered.reset_index(drop=True)


def min_value_filter(df: pd.DataFrame, min_dollar_value: float) -> pd.DataFrame:
    sym_med_close = df.groupby('symbol')[['close']].median()
    passed_sym = sym_med_close.loc[sym_med_close['close'] > min_dollar_value].index
    df_filtered = df.loc[df.symbol.isin(passed_sym)]
    return df_filtered.reset_index(drop=True)


def symbol_filter(df: pd.DataFrame, symbols: list):
    df_filtered = df.loc[df.symbol.isin(symbols), :]
    return df_filtered.reset_index(drop=True)


def outlier_squeeze(x, t: int=4):
    """A transformation that suppresses outliers for a standard normal."""
    xp = np.clip(x, -t, t)
    diff = np.tanh(x - xp)
    return xp + diff


def prepare_data(start_date: str, end_date: str) -> pd.DataFrame:
    
    df = get_dates_df(tick_type='daily', symbol='market', start_date=start_date, end_date=end_date)
    nrows_all = df.shape[0]
    print(nrows_all, 'Initial rows')
    
    df = all_dates_filer(df)
    nrows_1 = df.shape[0]
    print((nrows_1 - nrows_all), 'all dates filter')
     
    df = liquidity_filter(df, abs_dollar_cut=500_000)
    nrows_2 = df.shape[0]
    print((nrows_2 - nrows_1), 'liquidity filter')
    
    df = add_range(df)
    df = range_value_filter(df, low_cut=0.005, high_cut=0.5)
    nrows_3 = df.shape[0]
    print((nrows_3 - nrows_2), 'volitility filter')

    df = min_value_filter(df, min_dollar_value=1.0)
    nrows_4 = df.shape[0]
    print((nrows_4 - nrows_3), 'min $value filter')

    sym_details = pd.read_feather('data/sym_details.feather', columns=['symbol', 'name', 'sector', 'industry', 'tags', 'similar', 'type'])
    sym_labels = sym_details[(sym_details.sector!='') & (sym_details.type.str.upper()=='CS')].reset_index(drop=True)
    df = symbol_filter(df, symbols=sym_labels.symbol)
    nrows_5 = df.shape[0]
    print((nrows_5 - nrows_4), 'symbol details filter')
    
    # df = df.drop(columns=['date', 'midprice', 'range', 'dollar_total', 'range_value_pct'])
    df = df.drop(columns=['date', 'midprice', 'range'])
    df = df.set_index('date_time', drop=True)
    print(df.shape[0], 'Final rows', round(df.shape[0] / nrows_all, 3)*100, '% remaining')
    print(len(df.symbol.uniquie()), 'symbols included')
    
    # pivot df 'wide' by symbol
    m_close = df.pivot(columns='symbol', values='close')
    m_returns = m_close.diff().dropna()  # returns
    m_log_returns = pd.DataFrame(np.log(m_close)).diff().dropna() # log
    m_zs_returns = (m_log_returns - m_log_returns.mean()) / m_log_returns.std(ddof=0)  # z-score
    m_g_zs_returns = outlier_squeeze(m_zs_returns, t=4) # reduce outliners

    return df, sym_labels, m_close, m_returns, m_log_returns, m_zs_returns, m_g_zs_returns
