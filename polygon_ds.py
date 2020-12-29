import pandas as pd
from pyarrow.dataset import dataset, field
from pyarrow._dataset import FileSystemDataset
from utils_globals import LOCAL_PATH, S3_PATH, B2_ACCESS_KEY_ID, B2_SECRET_ACCESS_KEY, B2_ENDPOINT_URL


def get_s3_dataset(symbol: str, tick_type: str) -> FileSystemDataset:
    from pyarrow.fs import S3FileSystem
    s3  = S3FileSystem(
        access_key=B2_ACCESS_KEY_ID,
        secret_key=B2_SECRET_ACCESS_KEY,
        endpoint_override=B2_ENDPOINT_URL
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
    return ds.to_table(filter=filter_exp).to_pandas()
