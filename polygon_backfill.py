from os.path import join
from glob import glob
from pathlib import Path
import datetime
import pandas as pd
from pandas_market_calendars import get_calendar
from polygon_rest_api import get_grouped_daily, get_tickers_page, get_stock_ticks


def get_ticks_date(symbol: str, date: str, tick_type:str):
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

    return ticks


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
    # df['tick_dt'] = pd.to_datetime(df['tick_epoch'], utc=True, unit='ns')
    # df = df.sort_values(by=['tick_epoch', 'sequence'], ascending=True)
    return df


def get_open_market_dates(start_date, end_date):
    market = get_calendar('NYSE')
    schedule = market.schedule(start_date=start_date, end_date=end_date)
    dates = [i.date().isoformat() for i in schedule.index]
    return dates


def filename_dates_from_path(dates_path):
    if os.path.exists(dates_path):
        file_list = os.listdir(dates_path)
        # assumes {symbol}_{yyyy-mm-dd}.{format} filename template
        existing_dates = [i.split('_')[1].split('.')[0] for i in file_list]
        existing_dates.sort()
        output = existing_dates
    else:
        output = []
    return output


def find_remaining_dates(req_dates, existing_dates):
    existing_dates_set = set(existing_dates)
    remaining_dates = [x for x in req_dates if x not in existing_dates_set]
    next_dates = [i for i in remaining_dates if i <= datetime.date.today().isoformat()]
    print('pull new dates: ', next_dates)
    return next_dates


def get_market_candels_date(date='2020-01-02'):
    daily = get_grouped_daily(locale='us', market='stocks', date=date)
    daily_df = pd.DataFrame(daily)
    daily_df = daily_df.rename(columns={'T': 'symbol',
                                        'v': 'volume',
                                        'o': 'open',
                                        'c': 'close',
                                        'h': 'high',
                                        'l': 'low',
                                        't': 'time_epoch'})
    daily_df['date_time'] = pd.to_datetime(daily_df.time_epoch, utc=True, unit='ms') #.tz_convert('America/New_York')
    return daily_df


def save_df(df:pd.DataFrame, symbol:str, date:str, result_path:str):
    
    Path(f"{result_path}/csv/{symbol}").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{result_path}/csv/{symbol}/{symbol}_{date}.csv", index=False)
    
    Path(f"{result_path}/parquet/{symbol}").mkdir(parents=True, exist_ok=True)
    df.to_parquet(f"{result_path}/parquet/{symbol}/{symbol}_{date}.parquet", index=False, engine='fastparquet', compression='gzip')

    # Path(f"{result_path}/parquet_pyarrow/{symbol}").mkdir(parents=True, exist_ok=True)
    # df.to_parquet(f"{result_path}/parquet_pyarrow/{symbol}/{symbol}_{date}.parquet", index=False, engine='pyarrow', compression=compression)
    
    # Path(f"{result_path}/feather/{symbol}").mkdir(parents=True, exist_ok=True)
    # df.to_feather(f"{result_path}/feather/{symbol}/{symbol}_{date}.feather")


def validate_trades(trades_df):
    if len(trades_df) < 1:
        raise ValueError('0 length trades df')
    elif any(trades_df.count() == 0):
        raise ValueError('trades df missing fields. Recent historic data may not be ready for consumption')
    else:
        return(trades_df)


def save_trades_dates(symbol, start_date, end_date, result_path):
    req_dates = get_open_market_dates(start_date, end_date)
    existing_dates = filename_dates_from_path(f"{result_path}/csv/{symbol}")
    dates = find_remaining_dates(req_dates, existing_dates)
    for date in dates:
        trades = get_ticks_date(symbol, date, tick_type='trades')
        trades_df = trades_to_df(trades)
        trades_df = validate_trades(trades_df)
        save_df(trades_df, symbol, date, result_path)


def read_matching_files(glob_string, format='csv'):
    if format == 'csv':
        return pd.concat(map(pd.read_csv, glob(join('', glob_string))), ignore_index=True)
    elif format == 'parquet':
        return pd.concat(map(pd.read_parquet, glob(join('', glob_string))), ignore_index=True)
    elif format == 'feather':
        return pd.concat(map(pd.read_feather, glob(join('', glob_string))), ignore_index=True)
