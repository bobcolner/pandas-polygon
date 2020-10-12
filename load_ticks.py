import os
import pandas as pd
import s3_datasets as s3d
from local_backfill import ticks_df_tofile


def load_ticks(local_path:str, symbol:str, date:str, tick_type='trades', clean=True) -> pd.DataFrame:
    try:
        print('trying to get ticks from local file...')
        pd.read_feather(local_path + "{tick_type}/symbol={symbol}/date={date}/data.feather")
    except FileNotFoundError:
        try:
            print('trying to get ticks from s3...')
            df = s3d.get_tick_df(symbol, date, tick_type)
        except FileNotFoundError:
            print('trying to get ticks from polygon API')
            df = s3d.backfill_date_tos3(symbol, date, tick_type)
        finally:
            ticks_df_tofile(df, symbol, date, tick_type, local_path)
    
    if tick_type == 'trades' and clean and df:
        df = clean_trades_df(df)
        print(df.info())

    return df


def clean_trades_df(df:pd.DataFrame, small_df=True) -> pd.DataFrame:
    og_size = df.shape[0]
    # drop irrgular trade conditions
    df = df[df.irregular==False].reset_index(drop=True)
    
    # add dt diff
    dt_diff = (df.sip_dt - df.exchange_dt)
    
    # drop trades with >1sec timestamp diff
    df = df[dt_diff < pd.to_timedelta(1, unit='S')].reset_index(drop=True)
    
    # add median filter and remove outlier trades
    df['filter'] = df['price'].rolling(window=5, center=False, min_periods=1).median()
    df['filter_diff'] = abs(df['price'] - df['filter'])
    df['filter_pct'] = abs((1-(df['filter_diff'] / df['price'])))*100
    df['filter_zs'] = (df['filter_diff'] - df['filter_diff'].mean()) / df['filter_diff'].std(ddof=0)
    df = df[df.filter_zs < 10].reset_index(drop=True)
    
    # remove duplicate trades
    num_dups = sum(df.duplicated(subset=['sip_dt', 'exchange_dt', 'sequence', 'trade_id', 'price', 'size']))
    if num_dups > 0: 
        print(num_dups, ' duplicated trade removed')
        df = df.drop_duplicates(subset=['sip_dt', 'exchange_dt', 'sequence', 'trade_id', 'price', 'size'])
    
    # drop trades with zero size/volume
    df = df[df['size']>0].reset_index(drop=True)
    
    droped_rows = og_size - df.shape[0]
    print('dropped', droped_rows, 'ticks (', round((droped_rows/og_size)*100, 2), '%)')

    # sort df
    df = df.sort_values(['sip_dt', 'exchange_dt', 'sequence'])
    if small_df:
        df = df[['sip_dt', 'price', 'size']]
        # df['epoch'] = df.sip_dt.astype('int64')
        return df.rename(columns={'sip_dt': 'date_time', 'size': 'volume'})
    else:
        return df
