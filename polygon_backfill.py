from pathlib import Path
from tempfile import NamedTemporaryFile
import pandas as pd
from tqdm import tqdm
from polygon_rest_api import get_market_date, get_stocks_ticks_date
from polygon_s3 import get_s3fs_client, get_symbol_dates


def read_matching_files(glob_string: str, reader=pd.read_csv) -> pd.DataFrame:
    from glob import glob
    return pd.concat(map(reader, glob(path.join('', glob_string))), ignore_index=True)


def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError('df is NoneType')
    if len(df) < 1:
        raise ValueError('zero row df')
    elif any(df.count() == 0):
        raise ValueError('df has fields with no values. Recent historic data may not be ready for consumption')
    else:
        return df
 

def market_daily_to_df(daily: list) -> pd.DataFrame:    
    df = pd.DataFrame(daily, columns=['T', 'v', 'o', 'c', 'h', 'l', 'vw', 't'])
    df = df.rename(columns={'T': 'symbol',
                            'v': 'volume',
                            'o': 'open',
                            'c': 'close',
                            'h': 'high',
                            'l': 'low',
                           'vw': 'vwap',
                            't': 'epoch'})
    # add datetime index
    df['date_time'] = pd.to_datetime(df['epoch'] * 10**6).dt.normalize()
    # df = df.set_index(pd.to_datetime(df['epoch'] * 10**6).dt.normalize(), drop=True)
    # df = df.rename_axis(index='date')
    df = df.drop(columns='epoch')
    # fix vwap
    mask = ~(df.vwap.between(df.low, df.high)) # vwap outside the high/low range
    df.loc[mask, 'vwap'] = df.loc[mask, 'close'] # replace bad vwap with close price
    # add dollar total
    df['dollar_total'] = df['vwap'] * df['volume']
    # optimze datatypes
    df['volume'] = df['volume'].astype('uint64')
    for col in ['dollar_total', 'vwap', 'open', 'close', 'high', 'low']:
        df[col] = df[col].astype('float32')
    return df


def ticks_to_df(ticks: list, tick_type: str) -> pd.DataFrame:
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
    
    # cast datetimes (for both trades+quotes)
    df['sequence'] = df['sequence'].astype('uint32')
    df['sip_dt'] = pd.to_datetime(df['sip_epoch'], unit='ns')
    df['exchange_dt'] = pd.to_datetime(df['exchange_epoch'], unit='ns')
    # drop columns
    df = df.drop(columns=['tape', 'sip_epoch', 'exchange_epoch'])
    return df.reset_index(drop=True)


def med_filter(df: pd.DataFrame, window: int=5, zthresh: int=10) -> pd.DataFrame:
    df['filter'] = df['price'].rolling(window, center=False, min_periods=1).median()
    df['filter_diff'] = abs(df['price'] - df['filter'])
    df['filter_zs'] = (df['filter_diff'] - df['filter_diff'].mean()) / df['filter_diff'].std(ddof=0)
    return df.loc[df.filter_zs < zthresh].reset_index(drop=True)


def clean_trades_df(df: pd.DataFrame, small: bool=True) -> pd.DataFrame:
    # get origional number of ticks
    og_tick_count = df.shape[0]
    # drop irrgular trade conditions
    df = df.loc[df.irregular==False]
    # drop trades with >1sec timestamp diff
    dt_diff = (df.sip_dt - df.exchange_dt)
    df = df.loc[dt_diff < pd.to_timedelta(1, unit='S')]
    # add median filter and remove outlier trades
    df = med_filter(df, window=5, zthresh=10)
    # remove duplicate trades
    num_dups = sum(df.duplicated(subset=['sip_dt', 'exchange_dt', 'sequence', 'trade_id', 'price', 'size']))
    if num_dups > 0: 
        print(num_dups, 'duplicated trade removed')
        df = df.drop_duplicates(subset=['sip_dt', 'exchange_dt', 'sequence', 'trade_id', 'price', 'size'])
    # drop trades with zero size/volume
    df = df.loc[df['size'] > 0]
    droped_rows = og_tick_count - df.shape[0]
    print('dropped', droped_rows, 'ticks (', round((droped_rows / og_tick_count) * 100, 2), '%)')
    # sort df
    df = df.sort_values(['sip_dt', 'exchange_dt', 'sequence'])
    if small:
        df = df[['sip_dt', 'price', 'size']]
        return df.rename(columns={'sip_dt': 'date_time', 'size': 'volume'}).reset_index(drop=True)
    else:
        return df.reset_index(drop=True)


def get_ticks_date_df(symbol: str, date: str, tick_type: str='trades', clean: bool=True, small: bool=True) -> pd.DataFrame:
    ticks = get_stocks_ticks_date(symbol, date, tick_type)
    if len(ticks) < 1:
        return pd.DataFrame() # return empty df
    else:    
        df = ticks_to_df(ticks, tick_type)
    if tick_type == 'trades' and clean:
        df = clean_trades_df(df, small)
    return validate_df(df)


def get_market_date_df(date: str) -> pd.DataFrame:
    daily = get_market_date(locale='us', market='stocks', date=date)
    if len(daily) < 1:
        raise ValueError('get_market_date returned zero rows')
    return market_daily_to_df(daily)


def backfill_date(symbol: str, date: str, tick_type: str, result_path: str, save_local=True, upload_to_s3=False) -> pd.DataFrame:
    
    if upload_to_s3:
        s3fs = get_s3fs_client()

    if symbol == 'market':
        df = get_market_date_df(date)
    else: # get tick data
        df = get_ticks_date_df(symbol, date, tick_type)
        if len(df) < 1:
            print('No Data for', symbol, date)
            return df
    
    if tick_type is None:
        tick_type = 'daily'

    if save_local: # save to local file
        full_path = result_path + f"/{tick_type}/symbol={symbol}/date={date}/"
        Path(full_path).mkdir(parents=True, exist_ok=True)
        file_path = full_path + 'data.feather'
        print('Saving:', symbol, date, 'to local file')
        df.to_feather(path=file_path, version=2)
    else:
        with NamedTemporaryFile(mode='w+b') as tmp_ref1:
            file_path = tmp_ref1.name
            df.to_feather(path=file_path, version=2)
    
    if upload_to_s3: # upload to s3/b2
        print('Uploading:', symbol, date, 'to S3/B2')
        s3fs.put(file_path, f"polygon-equities/data/{tick_type}/symbol={symbol}/date={date}/data.feather")

    return df


def get_open_market_dates(start_date: str, end_date: str) -> list:
    from pandas_market_calendars import get_calendar
    market = get_calendar('NYSE')
    schedule = market.schedule(start_date=start_date, end_date=end_date)
    dates = [i.date().isoformat() for i in schedule.index]
    return dates


def dates_from_path(symbol: str, tick_type: str, result_path: str) -> list:
    from os import listdir
    # assumes 'hive' {date}={yyyy-mm-dd}/data.{format} filename template
    dates_path = f"{result_path}/{tick_type}/symbol={symbol}"
    if Path(dates_path).exists():    
        file_list = listdir(dates_path)
        if '.DS_Store' in file_list:
            file_list.remove('.DS_Store')
        existing_dates = [i.split('=')[1] for i in file_list]
    else:
        existing_dates = []
    return existing_dates


def find_remaining_dates(request_dates: str, existing_dates: str) -> list:
    from datetime import date
    existing_dates_set = set(existing_dates)
    remaining_dates = [x for x in request_dates if x not in existing_dates_set and x <= date.today().isoformat()]
    return remaining_dates


def backfill_dates(symbol: str, start_date: str, end_date: str, result_path: str, tick_type: str, save_local=True, upload_to_s3=True):
    
    request_dates = get_open_market_dates(start_date, end_date)
    print('requested', len(request_dates), 'dates')
    
    if upload_to_s3:
        existing_dates = get_symbol_dates(symbol, tick_type)
    else:
        existing_dates = dates_from_path(symbol, tick_type, result_path)

    if existing_dates is not None:
        request_dates = find_remaining_dates(request_dates, existing_dates)
    
    print(len(request_dates), 'remaining dates')
    
    for date in tqdm(request_dates):
        print('fetching:', date)
        backfill_date(symbol, date, tick_type, result_path, save_local, upload_to_s3)
