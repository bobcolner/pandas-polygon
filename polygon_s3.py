from os import environ
from io import BytesIO
from tempfile import NamedTemporaryFile
from pathlib import Path
import pandas as pd
from pyarrow.dataset import dataset, field
from pyarrow._dataset import FileSystemDataset


def get_s3fs_client(cached: bool=False):
    if cached:
        from fsspec import filesystem
        # https://filesystem-spec.readthedocs.io/
        s3fs = filesystem(
            protocol='filecache',
            target_protocol='s3',
            target_options={
                'key': environ['B2_ACCESS_KEY_ID'],
                'secret': environ['B2_SECRET_ACCESS_KEY'],
                'client_kwargs': {'endpoint_url': environ['B2_ENDPOINT_URL']}
                },
            # cache_storage='/Users/bobcolner/QuantClarity/pandas-polygon/data/cache'
            )
    else:
        from s3fs import S3FileSystem
        s3fs = S3FileSystem(
                key=environ['B2_ACCESS_KEY_ID'], 
                secret=environ['B2_SECRET_ACCESS_KEY'], 
                client_kwargs={'endpoint_url': environ['B2_ENDPOINT_URL']}
            )
    return s3fs


s3fs = get_s3fs_client(cached=False)


def get_symbol_dates(symbol: str, tick_type: str='trades') -> str:
    paths = s3fs.ls(path=f"polygon-equities/data/{tick_type}/symbol={symbol}/", refresh=True)
    return [path.split('date=')[1] for path in paths]


def get_symbols(tick_type: str='trades') -> str:
    paths = s3fs.ls(path=f"polygon-equities/data/{tick_type}/", refresh=True)
    return [path.split('symbol=')[1] for path in paths]


def remove_symbol(symbol: str, tick_type: str):
    path = f"polygon-equities/data/{tick_type}/symbol={symbol}/"
    s3fs.rm(path, recursive=True)

def find_symbol_storage_used(symbol: str, tick_type: str) -> dict:
    path = f"polygon-equities/data/{tick_type}/symbol={symbol}/"
    return s3fs.du(path)


def put_date_df_to_s3(symbol: str, date: str, tick_type: str) -> pd.DataFrame:
    df = get_ticks_date_df(symbol, date, tick_type)
    with NamedTemporaryFile(mode='w+b') as tmp_ref1:
        df.to_feather(path=tmp_ref1.name, version=2)
        s3fs.put(tmp_ref1.name, f"polygon-equities/data/{tick_type}/symbol={symbol}/date={date}/data.feather")
    return df


def get_date_df_from_s3(symbol: str, date: str, tick_type: str='trades', columns=None) -> pd.DataFrame:    
    byte_data = s3fs.cat(f"polygon-equities/data/{tick_type}/symbol={symbol}/date={date}/data.feather")
    if columns:
        df = pd.read_feather(BytesIO(byte_data), columns=columns)
    else:
        df = pd.read_feather(BytesIO(byte_data))
    return df


def date_df_to_file(df: pd.DataFrame, symbol:str, date:str, tick_type: str, result_path: str) -> str:
    path = result_path + f"{tick_type}/symbol={symbol}/date={date}/"
    Path(path).mkdir(parents=True, exist_ok=True)
    df.to_feather(path+'data.feather', version=2)
    return path+'data.feather'


def load_ticks(local_path:str, symbol:str, date:str, tick_type: str='trades') -> pd.DataFrame:
    try:
        print('trying to get ticks from local file...')
        df = pd.read_feather(local_path + f"/{tick_type}/symbol={symbol}/date={date}/data.feather")
    except FileNotFoundError:
        try:
            print('trying to get ticks from s3...')
            df = get_date_df_from_s3(symbol, date, tick_type)
        except FileNotFoundError:
            print('trying to get ticks from polygon API and save to s3...')
            df = put_date_df_to_s3(symbol, date, tick_type)
        finally:
            print('saving ticks to local file')
            path = date_df_to_file(df, symbol, date, tick_type, local_path)
    return df


def get_s3_dataset() -> FileSystemDataset:
    from pyarrow.fs import S3FileSystem
    s3  = S3FileSystem(
        access_key=environ['B2_ACCESS_KEY_ID'],
        secret_key=environ['B2_SECRET_ACCESS_KEY'],
        endpoint_override=environ['B2_ENDPOINT_URL']
    )
    ds = dataset(
        source='polygon-equities/data/trades/',
        format='feather',
        filesystem=s3,
        partitioning='hive',
        exclude_invalid_files=True
    )
    return ds


def get_local_dataset(result_path: str) -> FileSystemDataset:
    ds = dataset(
        source=result_path,
        format='feather',
        partitioning='hive',
        exclude_invalid_files=True
    )
    return ds


def dataset_to_df(ds: FileSystemDataset, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    
    filter_exp = (field('symbol') == symbol) & \
        (field('date') >= start_date) & (field('date') <= end_date)

    return ds.to_table(filter=filter_exp).to_pandas()
