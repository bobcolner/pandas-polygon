from os import environ
import pandas as pd
from pyarrow.dataset import dataset, field
from pyarrow._dataset import FileSystemDataset

LOCAL_PATH = environ['LOCAL_PATH']
S3_PATH = environ['S3_PATH']


# pyarrow datasets functions
def get_s3_dataset(symbol: str, tick_type: str='trades') -> FileSystemDataset:
    from pyarrow.fs import S3FileSystem

    s3  = S3FileSystem(
        access_key=environ['B2_ACCESS_KEY_ID'],
        secret_key=environ['B2_SECRET_ACCESS_KEY'],
        endpoint_override=environ['B2_ENDPOINT_URL']
    )
    ds = dataset(
        source=S3_PATH + f"/{tick_type}/symbol={symbol}/",
        format='feather',
        filesystem=s3,
        partitioning='hive',
        exclude_invalid_files=True
    )
    return ds


def get_local_dataset(tick_type: str, symbol: str=None) -> FileSystemDataset:

    full_path = LOCAL_PATH + f"/{tick_type}/"
    if symbol:
        full_path = full_path + f"symbol={symbol}/"
    ds = dataset(
        source=full_path,
        format='feather',
        partitioning='hive',
        exclude_invalid_files=True
    )
    return ds


def get_dates_df(symbol: str, tick_type: str, start_date: str, end_date: str, source: str='local') -> pd.DataFrame:
    if source == 'local':
        ds = get_local_dataset(tick_type=tick_type, symbol=symbol)
    elif source == 's3':
        ds = get_s3_dataset(tick_type=tick_type, symbol=symbol)
    filter_exp = (field('date') >= start_date) & (field('date') <= end_date)
    df = ds.to_table(filter=filter_exp).to_pandas()
    return df


# def get_symbol_trades_df(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
#     ds = get_local_dataset(tick_type='trades', symbol=symbol)
#     filter_exp = (field('date') >= start_date) & (field('date') <= end_date)
#     df = ds.to_table(filter=filter_exp).to_pandas()
#     return df


# def get_market_daily_df(start_date: str, end_date: str, symbol: str=None) -> pd.DataFrame:
#     ds = get_local_dataset(tick_type='daily', symbol='market')
#     filter_exp = (field('date') >= start_date) & (field('date') <= end_date)
#     df = ds.to_table(filter=filter_exp).to_pandas()
#     if symbol:
#         # df = ds.to_table(filter=field('symbol') == symbol).to_pandas()
#         df = df.loc[df['symbol'] == symbol]
#     return df.reset_index(drop=True)