from pathlib import Path
import datetime
import pandas as pd
from polygon_rest_api import get_grouped_daily
import polygon_backfill as pb


def ticks_df_tofile(df:pd.DataFrame, symbol:str, date:str, tick_type:str, result_path:str):
    # 'hive' folder template
    path = result_path + f"{tick_type}/symbol={symbol}/date={date}/"
    Path(path).mkdir(parents=True, exist_ok=True)
    df.to_feather(path+'data.feather', version=2)


def dates_from_path(dates_path:str) -> list:
    # assumes 'hive' {date}={yyyy-mm-dd}/data.{format} filename template
    if os.path.exists(dates_path):        
        file_list = os.listdir(dates_path)
        if '.DS_Store' in file_list:
            file_list.remove('.DS_Store')
        existing_dates = [i.split('=')[1] for i in file_list]
    else:
        existing_dates = []
    return existing_dates


def find_remaining_dates(req_dates:str, existing_dates:str) -> list:
    existing_dates_set = set(existing_dates)
    remaining_dates = [x for x in req_dates if x not in existing_dates_set]
    next_dates = [i for i in remaining_dates if i <= datetime.date.today().isoformat()]
    return next_dates


def backfill_dates_tofile(symbol:str, start_date:str, end_date:str, result_path:str, 
    tick_type:str, skip_existing=False) -> str:
    
    req_dates = pb.get_open_market_dates(start_date, end_date)
    print(len(req_dates), 'requested dates')

    if skip_existing == True: # only appies to tick data
        existing_dates = dates_from_path(f"{result_path}/{symbol}")
        if existing_dates is not None:
            req_dates = find_remaining_dates(req_dates, existing_dates)
        print(len(req_dates), 'remaining dates')
    
    for date in req_dates:
        print(date)
        if symbol == 'market_daily':
            daily = get_grouped_daily(locale='us', market='stocks', date=date)
            df = pb.get_market_daily_df(daily)
            full_result_path = result_path
        else: # get tick data
            df = pb.backfill_date_todf(symbol, date, tick_type)
            # dir_path = ticks_df_tofile(df, symbol, date, tick_type, result_path + f"/ticks/{tick_type}")
            dir_path = ticks_df_tofile(df, symbol, date, tick_type, result_path)
