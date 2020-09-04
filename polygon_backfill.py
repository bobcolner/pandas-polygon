import os
from io import BytesIO
from time import time_ns
from glob import glob
from pathlib import Path
import datetime
from tempfile import NamedTemporaryFile
import pandas as pd
from pandas_market_calendars import get_calendar
from polygon_rest_api import get_grouped_daily, get_stock_ticks
import s3fs


def read_market_daily(result_path:str) -> pd.DataFrame:    
    df = read_matching_files(glob_string=result_path+'/market_daily/*.feather', reader=pd.read_feather)
    df = df.set_index(pd.to_datetime(df.date), drop=True)
    df = df.drop(columns=['date'])
    df = find_compleat_symbols(df, compleat_only=True)
    df = df.sort_index()
    return df


def read_matching_files(glob_string:str, reader=pd.read_csv):
    return pd.concat(map(reader, glob(os.path.join('', glob_string))), ignore_index=True)


def trades_to_df(ticks:list) -> pd.DataFrame:
    df = pd.DataFrame(ticks, columns=['t', 'y', 'q', 'i', 'x', 'p', 's', 'c', 'z'])
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
    df['sequence'] = df['sequence'].astype('uint32')
    df['trade_id'] = df['trade_id'].astype('string')
    df['tape'] = df['tape'].astype('uint8')
    # df['tick_dt'] = pd.to_datetime(df['sip_epoch'], utc=True, unit='ns')
    # df = df.sort_values(by=['sip_epoch', 'sequence'], ascending=True)
    return df


def quotes_to_df(ticks:list) -> pd.DataFrame:
    df = pd.DataFrame(ticks, columns=['t', 'y', 'q', 'x', 'X', 'p', 'P', 's', 'S'])
    df = df.rename(columns={'p': 'bid_price',
                            'P': 'ask_price',
                            's': 'bid_size',
                            'S': 'ask_size',
                            'x': 'bid_exchange_id',
                            'X': 'ask_exchange_id',
                            't': 'sip_epoch',
                            'y': 'exchange_epoch',
                            'q': 'sequence'
                            })
    # optimze datatypes
    df['bid_price'] = df['bid_price'].astype('float32')
    df['ask_price'] = df['ask_price'].astype('float32')
    df['bid_size'] = df['bid_size'].astype('uint32')
    df['ask_size'] = df['ask_size'].astype('uint32')
    df['bid_exchange_id'] = df['bid_exchange_id'].astype('uint8')
    df['ask_exchange_id'] = df['ask_exchange_id'].astype('uint8')
    df['sequence'] = df['sequence'].astype('uint32')
    return df


def get_open_market_dates(start_date:str, end_date:str) -> list:
    market = get_calendar('NYSE')
    schedule = market.schedule(start_date=start_date, end_date=end_date)
    dates = [i.date().isoformat() for i in schedule.index]
    return dates


def dates_from_path(dates_path:str, date_partition:str) -> list:
    if os.path.exists(dates_path):
        file_list = os.listdir(dates_path)
        if '.DS_Store' in file_list:
            file_list.remove('.DS_Store')

        if date_partition == 'file_symbol_date':
            # assumes {symbol}_{yyyy-mm-dd}.{format} filename template
            existing_dates = [i.split('_')[1].split('.')[0] for i in file_list]
        
        elif date_partition == 'file_dates':
            # assumes {yyyy-mm-dd}.{format} filename template
            existing_dates = [i.split('.')[0] for i in file_list]
        
        elif date_partition == 'dir_dates':
            # assumes {yyyy-mm-dd}/data.{format} filename template
            existing_dates = file_list

        elif date_partition == 'hive':
            # assumes {date}={yyyy-mm-dd}/data.{format} filename template
            existing_dates = [i.split('=')[1] for i in file_list]

        return existing_dates


def find_remaining_dates(req_dates:str, existing_dates:str) -> list:
    existing_dates_set = set(existing_dates)
    remaining_dates = [x for x in req_dates if x not in existing_dates_set]
    next_dates = [i for i in remaining_dates if i <= datetime.date.today().isoformat()]
    return next_dates


def save_df(df:pd.DataFrame, symbol:str, date:str, result_path:str, 
    date_partition:str, formats=['parquet', 'feather']) -> str:

    if date_partition == 'file_dates':
        partion_path = f"{symbol}/{date}"
    elif date_partition == 'dir_dates':
        partion_path = f"{symbol}/{date}/"
    elif date_partition == 'hive':
        partion_path = f"symbol={symbol}/date={date}/"
    
    if 'csv' in formats:
        path = result_path + '/csv/' + partion_path
        Path(path).mkdir(parents=True, exist_ok=True)
        df.to_csv(
            path_or_buf=path+'data.csv',
            index=False,
        )
    if 'parquet' in formats:
        path = result_path + '/parquet/' + partion_path
        Path(path).mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            path=path+'data.parquet',
            engine='auto',
            # compression='brotli',  # 'snappy', 'gzip', 'brotli', None
            index=False,
            partition_cols=None,
        )
    if 'feather' in formats:
        path = result_path + '/feather/' + partion_path
        Path(path).mkdir(parents=True, exist_ok=True)
        df.to_feather(path+'data.feather', version=2)
    return dir_path


def validate_df(df):
    if df is None:
        raise ValueError('df is NoneType')
    if len(df) < 1:
        raise ValueError('0 length df')
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


def get_ticks_date(symbol: str, date: str, tick_type:str) -> list:
    last_tick = None
    limit = 50000
    ticks = []
    run = True
    while run == True:
        # get batch of ticks
        ticks_batch = get_stock_ticks(symbol, date, tick_type, timestamp_first=last_tick, limit=limit)
        # update last_tick
        if len(ticks_batch) < 1:  # empty tick batch
            run = False
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


def backfill_dates_tofile(symbol:str, start_date:str, end_date:str, result_path:str, 
    date_partition:str, tick_type:str, formats=['feather'], skip=False) -> str:
    
    req_dates = get_open_market_dates(start_date, end_date)
    print(len(req_dates), 'requested dates')

    if skip == True: # only appies to tick data
        existing_dates = dates_from_path(f"{result_path}/{symbol}", date_partition)
        if existing_dates is not None:
            req_dates = find_remaining_dates(req_dates, existing_dates)
        print(len(req_dates), 'remaining dates')
    
    for date in req_dates:
        print(date)
        if symbol == 'market_daily':
            daily = get_grouped_daily(locale='us', market='stocks', date=date)
            df = get_market_daily_df(daily)
            full_result_path = result_path

        if tick_type == 'trades':
            trade_ticks = get_ticks_date(symbol, date, tick_type='trades')
            df = trades_to_df(trade_ticks)
            full_result_path = result_path + '/ticks/trades'
        elif tick_type == 'quotes':
            quote_ticks = get_ticks_date(symbol, date, tick_type='quotes')
            df = quotes_to_df(quote_ticks)
            full_result_path = result_path + '/ticks/quotes'

        df = validate_df(df)
        save_df(df, symbol, date, full_result_path, date_partition, formats)

    return full_result_path


def backfill_date_todf(symbol:str, date:str, tick_type:str) -> pd.DataFrame:

    if symbol == 'market_daily':
        daily = get_grouped_daily(locale='us', market='stocks', date=date)
        df = get_market_daily_df(daily)

    if tick_type == 'trades':
        trade_ticks = get_ticks_date(symbol, date, tick_type)
        df = trades_to_df(trade_ticks)
    elif tick_type == 'quotes':
        quote_ticks = get_ticks_date(symbol, date, tick_type)
        df = quotes_to_df(quote_ticks)

    return validate_df(df)


def backfill_date_tofile(symbol:str, date:str, tick_type:str, result_path:str) -> bool:
    df = backfill_date_todf(symbol, date, tick_type)
    dir_path = save_df(df, symbol, date, result_path+f"/{tick_type}", 
        date_partition='hive', formats=['parquet','feather','csv'])
    return True


s3 = s3fs.S3FileSystem(
        key=os.environ['B2_ACCESS_KEY_ID'], 
        secret=os.environ['B2_SECRET_ACCESS_KEY'], 
        client_kwargs={'endpoint_url': os.environ['B2_ENDPOINT_URL']}
    )

def backfill_date_tos3(symbol:str, date:str, tick_type:str) -> bool:
    df = backfill_date_todf(symbol, date, tick_type)    
    with NamedTemporaryFile(mode='w+b') as tmp_ref1:
        df.to_feather(path=tmp_ref1.name, version=2)
        s3.put(tmp_ref1.name, f"polygon-equities/data/{tick_type}/symbol={symbol}/date={date}/data.feather")
    
    return True


s3 = s3fs.S3FileSystem(
        key=os.environ['B2_ACCESS_KEY_ID'], 
        secret=os.environ['B2_SECRET_ACCESS_KEY'], 
        client_kwargs={'endpoint_url': os.environ['B2_ENDPOINT_URL']}
    )

def list_s3_symbol(symbol:str) -> str:
    return s3.ls(f"polygon-equities/data/trades/symbol={symbol}/")

    
def get_s3_df(symbol:str, date:str, dt=True, columns=None) -> pd.DataFrame:
    byte_data = s3.cat(f"polygon-equities/data/trades/symbol={symbol}/date={date}/data.feather")
    df_bytes_io = BytesIO(byte_data)
    df = pd.read_feather(df_bytes_io, columns=columns)
    if dt:
        df['date_time'] = pd.to_datetime(df.exchange_epoch)
        df = df.drop(columns='exchange_epoch')
    return df


def condition_filter(condition_array, remove_condition=None, afterhours=False):
    remove_condition = [2, 5, 7, 10, 12, 13, 15, 16, 17, 18, 19, 20, 22, 28, 29, 33, 38, 52, 53]
    if afterhours is True:
        remove_condition.pop(21)
    if condition_array is not None:
        filter_ticks = any(np.isin(condition_array, remove_condition))
    else: 
        filter_ticks = False
    return filter_ticks


def add_price_outlier(df, window_len):
    med_smooth = df.price.rolling(window_len, center=True).median()
    df['outlier_diff'] = abs(df.price - med_smooth)
    df['outlier_pct'] = abs((1-(df.price / med_smooth)))*100
    df['outlier_zs'] = (df['outlier_diff'] - df['outlier_diff'].mean()) / df['outlier_diff'].std(ddof=0)
    return df


def apply_condition_filter_medfilter(df, window_len=7):
    bad_ticks = df.condition.apply(condition_filter)
    clean_df = df[~bad_ticks].reset_index(drop=True)
    clean_df = pb.add_price_outlier(clean_df, window_len)
    return clean_df, bad_ticks
    