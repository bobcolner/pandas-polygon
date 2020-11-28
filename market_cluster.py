import numpy as np
import pandas as pd
from corex_linearcorex import Corex
from polygon_ds import get_dates_df


def plot_daily_symbols(df: pd.DataFrame, symbols: list=['SPY', 'QQQ'], metric: str='close') -> pd.DataFrame:
    fdf = df[['symbol', metric]][df.symbol.isin(symbols)]
    pdf = fdf.pivot(columns='symbol', values=metric)
    pdf.plot_bokeh(kind='line', sizing_mode="scale_height", rangetool=True, title=str(symbols), ylabel=metric+' [$]', number_format="1.00 $")
    return pdf


def all_dates_filer(df: pd.DataFrame) -> pd.DataFrame:
    sym_count = df.groupby('symbol').count()[['open']]
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


def volitility_filter(df: pd.DataFrame, low_cut: float, high_cut: float) -> pd.DataFrame:
    sym_pct_range_med = df.groupby('symbol')[['range_value_pct']].median()
    passed_sym = sym_pct_range_med.loc[sym_pct_range_med['range_value_pct'].between(low_cut, high_cut)].index
    df_filtered = df.loc[df.symbol.isin(passed_sym)]
    return df_filtered.reset_index(drop=True)


def symbol_pivot(df: pd.DataFrame, col: str='close', output: str='returns') -> pd.DataFrame:
    # pivot symbols to columns
    close_prices = df.pivot(columns='symbol', values=col)
    if output == 'identity':
        return close_prices
    elif output in ('returns', 'zscore'):
        # get returns from price ts
        returns = close_prices.diff().drop(close_prices.index[0])  # drop NA first row
        if output == 'zscore':
            returns = (returns - returns.mean()) / returns.std(ddof=0)

        return returns


def market_cluster_workflow(start_date: str, end_date: str) -> pd.DataFrame:
    
    df = get_dates_df(tick_type='daily', symbol='market', start_date=start_date, end_date=end_date)
    nrows_all = df.shape[0]
    print(nrows_all)
    
    df = all_dates_filer(df)
    print((df.shape[0] - nrows_all) / nrows_all)
    
    df = liquidity_filter(df, abs_dollar_cut=500_000)
    print((df.shape[0] - nrows_all) / nrows_all)
    
    df = add_range(df)
    df = volitility_filter(df, low_cut=0.005, high_cut=0.2)
    print((df.shape[0] - nrows_all) / nrows_all)
    
    df = df.drop(columns=['date', 'high', 'low', 'open', 'midprice', 'range'])
    df = df.set_index('date_time', drop=True)
    
    df = symbol_pivot(df, col='close', output='returns')
    return df


def corex_fit(X: pd.DataFrame, n_hidden: int) -> tuple:
    corex = Corex(n_hidden=n_hidden, gaussianize='outliers', verbose=True)
    corex.fit(X)
    sym_clust = pd.DataFrame(list(zip(X.columns, corex.clusters()))).rename(columns={0: 'symbol', 1: 'cluster'})
    full_df = pd.merge(sym_clust, pd.Series(data=corex.tcs, name='tcs'), left_on='cluster', right_index=True)
    return corex, full_df
