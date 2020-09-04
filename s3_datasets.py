import os
from io import BytesIO
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import s3fs
from polygon
from polygon_backfill import backfill_date_todf


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


def list_s3_symbol(symbol:str) -> str:
    return s3.ls(f"polygon-equities/data/trades/symbol={symbol}/")

    
def get_s3_df(symbol:str, date:str, cast_datetime=True, columns=None, condition_filter=True,
    keep_afterhours=True, outlier_score=True) -> pd.DataFrame:
    byte_data = s3.cat(f"polygon-equities/data/trades/symbol={symbol}/date={date}/data.feather")
    df_bytes_io = BytesIO(byte_data)
    df = pd.read_feather(df_bytes_io, columns=columns)
    if cast_datetime:
        df['date_time'] = pd.to_datetime(df.exchange_epoch)
        df = df.drop(columns='exchange_epoch')
    if condition_filter:
        df = apply_condition_filter(df, keep_afterhours=True)
    if outlier_score:
        df = add_price_outlier(df, window_len=7)
    return df


def condition_filter(condition_array, keep_afterhours=True):
    remove_condition = [2, 5, 7, 10, 12, 13, 15, 16, 17, 18, 19, 20, 22, 28, 29, 33, 38, 52, 53]
    if keep_afterhours is False:
        remove_condition.remove(12)
    if condition_array is not None:
        filter_idx = any(np.isin(condition_array, remove_condition))
    else: 
        filter_idx = False
    return filter_idx


def add_price_outlier(df, window_len=7):
    med_smooth = df['price'].rolling(window_len, center=True).median()
    df['outlier_diff'] = abs(df['price'] - med_smooth)
    df['outlier_pct'] = abs((1-(df['price'] / med_smooth)))*100
    df['outlier_zs'] = (df['outlier_diff'] - df['outlier_diff'].mean()) / df['outlier_diff'].std(ddof=0)
    return df


def apply_condition_filter(df, keep_afterhours=True):
    bad_ticks = df.condition.apply(condition_filter, keep_afterhours=keep_afterhours)
    clean_df = df[~bad_ticks].reset_index(drop=True)
    return clean_df, bad_ticks
