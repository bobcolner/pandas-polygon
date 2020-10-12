import os
from glob import glob
import numpy as np
import pandas as pd
from pandas_market_calendars import get_calendar
from polygon_rest_api import get_grouped_daily, get_stock_ticks


def read_market_daily(result_path:str) -> pd.DataFrame:    
    df = read_matching_files(glob_string=result_path+'/market_daily/*.feather', reader=pd.read_feather)
    df = df.set_index(pd.to_datetime(df.date), drop=True)
    df = df.drop(columns=['date'])
    df = find_compleat_symbols(df, compleat_only=True)
    df = df.sort_index()
    return df


def read_matching_files(glob_string:str, reader=pd.read_csv) -> pd.DataFrame:
    return pd.concat(map(reader, glob(os.path.join('', glob_string))), ignore_index=True)


def get_open_market_dates(start_date:str, end_date:str) -> list:
    market = get_calendar('NYSE')
    schedule = market.schedule(start_date=start_date, end_date=end_date)
    dates = [i.date().isoformat() for i in schedule.index]
    return dates


def validate_df(df:pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError('df is NoneType')
    if len(df) < 1:
        raise ValueError('zero row df')
    elif any(df.count() == 0):
        raise ValueError('df has fields with no values. Recent historic data may not be ready for consumption')
    else:
        return df


def find_compleat_symbols(df:pd.DataFrame, active_days:int=1, compleat_only:bool=True) -> pd.DataFrame:
    # count aviables days for each symbol
    sym_count = df.groupby('symbol').count()['open']
    print(df.symbol.value_counts().describe())
    if compleat_only is True:
        active_days = max(df.symbol.value_counts())
    # filter symbols for active_days thresholdqa
    passed_sym = sym_count[sym_count >= active_days].index
    df_filtered = df[df.symbol.isin(passed_sym)]
    return df_filtered
 

def get_market_daily_df(daily:list) -> pd.DataFrame:
    
    if len(daily) < 1:
        raise ValueError('get_grouped_daily() returned zero rows')

    df = pd.DataFrame(daily, columns=['T', 'v', 'o', 'c', 'h', 'l', 'vw'])
    df = df.rename(columns={'T': 'symbol',
                            'v': 'volume',
                            'o': 'open',
                            'c': 'close',
                            'h': 'high',
                            'l': 'low',
                           'vw': 'vwap',
                            'n': 'count',
                            't': 'open_epoch'})
    # add columns
    df['date'] = date
    df['date'] = df['date'].astype('string')
    df['dollar_total'] = df['vwap'] * df['volume']
    # optimze datatypes
    df['volume'] = df['volume'].astype('uint64')
    for col in ['dollar_total', 'vwap', 'open', 'close', 'high', 'low']:
        df[col] = df[col].astype('float32')
    # filter low liquidity stocks
    df = df[df['dollar_total'] > 10 ** 4]
    return df


def add_cond_filter(ticks: list) -> list:
    green_conditions = [0, 1, 3, 4, 8, 9, 11, 14, 23, 25, 27, 28, 30, 36, 41]
    irregular_conditions = [2, 5, 7, 10, 13, 15, 16, 20, 21, 22, 29, 33, 38, 52, 53]
    blank_conditions = [6, 17, 18, 19, 24, 26, 32, 35, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 54, 55, 56, 57]
    for idx, tick in enumerate(ticks):
        if 'c' in tick:
            ticks[idx]['green'] = any(np.isin(tick['c'], green_conditions))
            ticks[idx]['irregular'] = any(np.isin(tick['c'], irregular_conditions))
            ticks[idx]['blank'] = any(np.isin(tick['c'], blank_conditions))
            ticks[idx]['afterhours'] = any(np.isin(tick['c'], 12))
            ticks[idx]['odd_lot'] = any(np.isin(tick['c'], 37))
        else:
            ticks[idx]['green'] = False
            ticks[idx]['irregular'] = False
            ticks[idx]['blank'] = True
            ticks[idx]['afterhours'] = False
            ticks[idx]['odd_lot'] = False
    return ticks


def get_ticks_date(symbol: str, date: str, tick_type:str) -> list:
    last_tick = None
    limit = 50000
    ticks = []
    run = True
    while run == True:
        # get batch of ticks
        ticks_batch = get_stock_ticks(symbol, date, tick_type, timestamp_first=last_tick, limit=limit)
        # filter ticks
        ticks_batch = add_cond_filter(ticks_batch)
        # update last_tick
        if len(ticks_batch) < 1:  # empty tick batch
            run = False
        last_tick = ticks_batch[-1]['y'] # exchange ts
        # logging
        last_tick_time = pd.to_datetime(last_tick, utc=True, unit='ns').tz_convert('America/New_York')
        print('Downloaded: ', len(ticks_batch), symbol, 'ticks; latest time(NYC): ', last_tick_time)
        # append batch to ticks list
        ticks = ticks + ticks_batch
        # check if we are done pulling ticks
        if len(ticks_batch) < limit:
            run = False
        elif len(ticks_batch) == limit:
            del ticks[-1] # drop last row to avoid dups

    return ticks


def ticks_to_df(ticks:list, tick_type:str) -> pd.DataFrame:
    if tick_type == 'trades':
        df = pd.DataFrame(ticks, columns=['t', 'y', 'q', 'i', 'x', 'p', 's', 'c', 'z', 'green', 'irregular', 'blank', 'afterhours'])
        df = df.rename(columns={'p': 'price',
                                's': 'size',
                                'x': 'exchange_id',
                                't': 'sip_epoch',
                                'y': 'exchange_epoch',
                                'q': 'sequence',
                                'i': 'trade_id',
                                'c': 'condition',
                                'z': 'tape'
                                })
        # optimize datatypes
        df['price'] = df['price'].astype('float32')
        df['size'] = df['size'].astype('uint32')
        df['exchange_id'] = df['exchange_id'].astype('uint8')
        df['trade_id'] = df['trade_id'].astype('string')
        df['green'] = df['green'].astype('bool')
        df['irregular'] = df['irregular'].astype('bool')
        df['blank'] = df['blank'].astype('bool')
        df['afterhours'] = df['afterhours'].astype('bool')

    elif tick_type == 'quotes':
        df = pd.DataFrame(ticks, columns=['t', 'y', 'q', 'x', 'X', 'p', 'P', 's', 'S', 'z'])
        df = df.rename(columns={'p': 'bid_price',
                                'P': 'ask_price',
                                's': 'bid_size',
                                'S': 'ask_size',
                                'x': 'bid_exchange_id',
                                'X': 'ask_exchange_id',
                                't': 'sip_epoch',
                                'y': 'exchange_epoch',
                                'q': 'sequence',
                                'z': 'tape'
                                })
        # optimze datatypes
        df['bid_price'] = df['bid_price'].astype('float32')
        df['ask_price'] = df['ask_price'].astype('float32')
        df['bid_size'] = df['bid_size'].astype('uint32')
        df['ask_size'] = df['ask_size'].astype('uint32')
        df['bid_exchange_id'] = df['bid_exchange_id'].astype('uint8')
        df['ask_exchange_id'] = df['ask_exchange_id'].astype('uint8')
    
    # cast datetimes
    df['sequence'] = df['sequence'].astype('uint32')
    df['sip_dt'] = pd.to_datetime(df['sip_epoch'], unit='ns')
    df['exchange_dt'] = pd.to_datetime(df['exchange_epoch'], unit='ns')
    # drop columns
    df = df.drop(columns='tape')
    df = df.drop(columns='sip_epoch')
    df = df.drop(columns='exchange_epoch')
    return df


def backfill_date_todf(symbol:str, date:str, tick_type:str) -> pd.DataFrame:
    if symbol == 'market_daily':
        daily = get_grouped_daily(locale='us', market='stocks', date=date)
        df = get_market_daily_df(daily)
    else:
        ticks = get_ticks_date(symbol, date, tick_type)
        df = ticks_to_df(ticks, tick_type)    
    return validate_df(df)
