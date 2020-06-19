import os
from glob import glob
from pathlib import Path
import datetime
import pandas as pd
from pandas_market_calendars import get_calendar
from polygon_rest_api import get_grouped_daily, get_stock_ticks


def read_matching_files(glob_string, reader=pd.read_csv):
    return pd.concat(map(reader, glob(os.path.join('', glob_string))), ignore_index=True)


def trades_to_df(ticks):
    df = pd.DataFrame(ticks, columns=['t', 'y', 'q', 'i', 'x', 'p', 's'])
    df = df.rename(columns={'p': 'price',
                            's': 'size',
                            'x': 'exchange_id',
                            't': 'tick_epoch',
                            'y': 'exchange_epoch',
                            'q': 'sequence',
                            'i': 'trade_id'
                            })
    # optimize datatypes
    df['price'] = pd.to_numeric(df['price'], downcast='float')
    cols = ['size', 'exchange_id', 'sequence']
    df[cols] = df[cols].apply(pd.to_numeric, downcast='unsigned')
    df['trade_id'] = df['trade_id'].astype('string')
    # df['tick_dt'] = pd.to_datetime(df['tick_epoch'], utc=True, unit='ns')
    # df = df.sort_values(by=['tick_epoch', 'sequence'], ascending=True)
    return df


def quotes_to_df(ticks):
    df = pd.DataFrame(ticks, columns=['t', 'y', 'q', 'x', 'X', 'p', 'P', 's', 'S'])
    df = df.rename(columns={'p': 'bid_price',
                            'P': 'ask_price',
                            's': 'bid_size',
                            'S': 'ask_size',
                            'x': 'bid_exchange_id',
                            'X': 'ask_exchange_id',
                            't': 'tick_epoch',
                            'y': 'exchange_epoch',
                            'q': 'sequence'
                            })
    # optimze datatypes
    cols = ['bid_size', 'ask_price']
    df[cols] = df[cols].apply(pd.to_numeric, downcast='float')
    cols = ['bid_size', 'ask_size', 'bid_exchange_id', 'ask_exchange_id', 'sequence']
    df[cols] = df[cols].apply(pd.to_numeric, downcast='unsigned')
    return df


def get_open_market_dates(start_date, end_date):
    market = get_calendar('NYSE')
    schedule = market.schedule(start_date=start_date, end_date=end_date)
    dates = [i.date().isoformat() for i in schedule.index]
    return dates


def dates_from_path(dates_path, date_partition):
    if os.path.exists(dates_path):
        
        file_list = os.listdir(dates_path)
        if '.DS_Store' in file_list:
            file_list.remove('.DS_Store')

        if date_partition == 'file_symbol_date':
            # assumes {symbol}_{yyyy-mm-dd}.{format} filename template
            existing_dates = [i.split('_')[1].split('.')[0] for i in file_list]
        
        elif date_partition == 'file_date':
            # assumes {yyyy-mm-dd}.{format} filename template
            existing_dates = [i.split('.')[0] for i in file_list]
        
        elif date_partition == 'dir_date':
            # assumes {yyyy-mm-dd}/data.{format} filename template
            existing_dates = file_list

        elif date_partition == 'hive':
            # assumes {date}={yyyy-mm-dd}/data.{format} filename template
            existing_dates = [i.split('=')[1] for i in file_list]

    return existing_dates


def find_remaining_dates(req_dates, existing_dates):
    existing_dates_set = set(existing_dates)
    remaining_dates = [x for x in req_dates if x not in existing_dates_set]
    next_dates = [i for i in remaining_dates if i <= datetime.date.today().isoformat()]
    print('pull new dates: ', next_dates)
    return next_dates


def save_df(df:pd.DataFrame, symbol:str, date:str, result_path:str, date_partition:str):

    if date_partition == 'file_date':
        Path(f"{result_path}/{symbol}").mkdir(parents=True, exist_ok=True)
        path = f"{result_path}/{symbol}/{date}"
    elif date_partition == 'dir_date':
        Path(f"{result_path}/{symbol}/{date}").mkdir(parents=True, exist_ok=True)
        path = f"{result_path}/{symbol}/{date}/data"
    
    df.to_csv(
        path + '.csv',
        index=False
    )
    
    df.to_parquet(
        path=path + '.parquet',
        engine='auto',
        compression='brotli', #{'snappy', 'gzip', 'brotli', None}
        index=False,
        partition_cols=None
    )

    # from pyarrow import fs
    # s3  = fs.S3FileSystem(region="us-east-2", creds=...)
    import pyarrow.feather as pf
    pf.write_feather(
        df,
        dest=path + '.feather',
        version=2,
        compression='zstd', # 'lz4'
        compression_level=None
    )


def validate_ticks(df):
    if len(df) < 1:
        raise ValueError('0 length trades df')
    elif any(df.count() == 0):
        raise ValueError('trades df missing fields. Recent historic data may not be ready for consumption')
    else:
        return df


def get_market_daily_df(date:str, sort=False):
    daily = get_grouped_daily(locale='us', market='stocks', date=date)
    df = pd.DataFrame(daily)
    df = df.rename(columns={'T': 'symbol',
                            'v': 'volume',
                            'o': 'open',
                            'c': 'close',
                            'h': 'high',
                            'l': 'low',
                            'vw': 'vwap',
                            'n': 'count',
                            't': 'open_epoch'})
    # add date column
    df['date'] = date
    df['date'] = df['date'].astype('string')
    # remove count
    df = df.drop(columns='count')
    df = df.drop(columns='open_epoch')
    # optimze datatypes
    df['dollar_total'] = df['vwap'] * df['volume']
    cols = ['dollar_total', 'vwap', 'open', 'close', 'high', 'low']
    df[cols] = df[cols].apply(pd.to_numeric, downcast='float')
    df['volume'] = df['volume'].astype('int')
    # filter stocks
    df = df[df['dollar_total'] > 10 ** 4]
    # df = df[df['vwap'] > 0.05]
    if sort == True:
        df = df.sort_values('dollar_total', ascending=False)
        df = df.reset_index(drop=True)
    return df


def get_ticks_df(symbol: str, date: str, tick_type:str):
    last_tick = None
    limit = 50000
    ticks = []
    run = True
    while run == True:
        # get batch of ticks
        ticks_batch = get_stock_ticks(symbol, date, tick_type, timestamp_first=last_tick, limit=limit)
        # update last_tick
        last_tick = ticks_batch[-1]['t']
        # logging
        last_tick_time = pd.to_datetime(last_tick, utc=True, unit='ns').tz_convert('America/New_York')
        print('Downloaded: ', len(ticks_batch), symbol, 'ticks; latest time(NYC): ', last_tick_time)
        # append batch to ticks list
        ticks = ticks + ticks_batch
        # check if we are down pulling ticks
        if len(ticks_batch) < limit:
            run = False
        elif len(ticks_batch) == limit:
            del ticks[-1] # drop last row to avoid dups
    
    if tick_type == 'trades':
        df = trades_to_df(ticks)
    elif tick_type == 'quotes':
        df =  quotes_to_df(ticks)
    
    return df


def backfill_dates(symbol, start_date, end_date, result_path, date_partition, tick_type, skip=False):
    
    req_dates = get_open_market_dates(start_date, end_date)
    
    if skip == True:
        existing_dates = dates_from_path(f"{result_path}/{symbol}", date_partition)
        req_dates = find_remaining_dates(req_dates, existing_dates)
    
    for date in req_dates:
        
        if symbol == 'market_daily':
            df = get_market_daily_df(date)
        else:    
            if tick_type == 'tick_trades':
                df = get_ticks_df(symbol, date, tick_type='trades')
                result_path = result_path + '/ticks/trades'

            elif tick_type == 'tick_quotes':
                df = get_ticks_df(symbol, date, tick_type='quotes')
                result_path = result_path + '/ticks/quotes'

            df = validate_ticks(df)

        save_df(df, symbol, date, result_path, date_partition)
        
