from os import environ
from pathlib import Path
import pandas as pd


LOCAL_PATH = environ['LOCAL_PATH']


def get_open_market_dates(start_date: str, end_date: str) -> list:
    from pandas_market_calendars import get_calendar
    market = get_calendar('NYSE')
    schedule = market.schedule(start_date=start_date, end_date=end_date)
    dates = [i.date().isoformat() for i in schedule.index]
    return dates


def list_dates_from_path(symbol: str, tick_type: str) -> list:
    from os import listdir
    # assumes 'hive' {date}={yyyy-mm-dd}/data.{format} filename template
    dates_path = f"{LOCAL_PATH}/{tick_type}/symbol={symbol}"
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


# def backfill_date(symbol: str, date: str, tick_type: str, save_local=True, upload_to_s3=False) -> pd.DataFrame:
# from tempfile import NamedTemporaryFile
# from polygon_s3 import get_s3fs_client
    
#     if upload_to_s3:
#         s3fs = get_s3fs_client()

#     if symbol == 'market':
#         df = get_market_date_df(date)
#         tick_type = 'daily'
#     else: # get tick data
#         try:
#             df = get_ticks_date_df(symbol, date, tick_type, clean=False)
#         except:
#             print('No Data for', symbol, date)
#             return pd.DataFrame()

#     if save_local: # save to local file
#         full_path = LOCAL_PATH + f"/{tick_type}/symbol={symbol}/date={date}/"
#         Path(full_path).mkdir(parents=True, exist_ok=True)
#         file_path = full_path + 'data.feather'
#         print('Saving:', symbol, date, 'to local file')
#         df.to_feather(path=file_path, version=2)
#     else:
#         with NamedTemporaryFile(mode='w+b') as tmp_ref1:
#             file_path = tmp_ref1.name
#             df.to_feather(path=file_path, version=2)
    
#     if upload_to_s3: # upload to s3/b2
#         print('Uploading:', symbol, date, 'to S3/B2')
#         s3fs.put(file_path, S3_PATH + f"/{tick_type}/symbol={symbol}/date={date}/data.feather")

#     return df
