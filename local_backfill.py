from pathlib import Path
import datetime
import pandas as pd
from polygon_rest_api import get_grouped_daily
import polygon_backfill as pb


def backfill_date_tofile(symbol:str, date:str, tick_type:str, result_path:str):
    df = pb.backfill_date_todf(symbol, date, tick_type)
    dir_path = save_df(df, symbol, date, result_path+f"/{tick_type}",
        date_partition='hive', formats=['feather'])


def save_df(df:pd.DataFrame, symbol:str, date:str, result_path:str, 
    date_partition:str, formats=['parquet', 'feather']):

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


def backfill_dates_tofile(symbol:str, start_date:str, end_date:str, result_path:str, 
    date_partition:str, tick_type:str, formats=['feather'], skip_existing=False) -> str:
    
    req_dates = pb.get_open_market_dates(start_date, end_date)
    print(len(req_dates), 'requested dates')

    if skip_existing == True: # only appies to tick data
        existing_dates = dates_from_path(f"{result_path}/{symbol}", date_partition)
        if existing_dates is not None:
            req_dates = find_remaining_dates(req_dates, existing_dates)
        print(len(req_dates), 'remaining dates')
    
    for date in req_dates:
        print(date)
        if symbol == 'market_daily':
            daily = get_grouped_daily(locale='us', market='stocks', date=date)
            df = pb.get_market_daily_df(daily)
            full_result_path = result_path
        else:
            # backfill_date_tofile(symbol, date:str, tick_type, result_path)
            ticks = pb.get_ticks_date(symbol, date, tick_type)
            df = pb.ticks_to_df(ticks, tick_type)
            full_result_path = result_path + f"/ticks/{tick_type}"
            df = pb.validate_df(df)
            save_df(df, symbol, date, full_result_path, date_partition, formats)

