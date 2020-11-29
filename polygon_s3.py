from io import BytesIO
from tempfile import NamedTemporaryFile
from pathlib import Path
import pandas as pd
from utils_globals import LOCAL_PATH, S3_PATH, B2_ACCESS_KEY_ID, B2_SECRET_ACCESS_KEY, B2_ENDPOINT_URL
from polygon_df import get_date_df



def get_s3fs_client(cached: bool=False):
    if cached:
        from fsspec import filesystem
        # https://filesystem-spec.readthedocs.io/
        s3fs = filesystem(
            protocol='filecache',
            target_protocol='s3',
            target_options={
                'key': B2_ACCESS_KEY_ID,
                'secret': B2_SECRET_ACCESS_KEY,
                'client_kwargs': {'endpoint_url': B2_ENDPOINT_URL}
                },
            # cache_storage='/Users/bobcolner/QuantClarity/pandas-polygon/data/cache'
            )
    else:
        from s3fs import S3FileSystem
        s3fs = S3FileSystem(
                key=B2_ACCESS_KEY_ID, 
                secret=B2_SECRET_ACCESS_KEY, 
                client_kwargs={'endpoint_url': B2_ENDPOINT_URL}
            )
    return s3fs


s3fs = get_s3fs_client(cached=False)


def list_symbol_dates(symbol: str, tick_type: str) -> str:
    paths = s3fs.ls(path=S3_PATH + f"/{tick_type}/symbol={symbol}/", refresh=True)
    return [path.split('date=')[1] for path in paths]


def list_symbols(tick_type: str) -> str:
    paths = s3fs.ls(path=S3_PATH + f"/{tick_type}/", refresh=True)
    return [path.split('symbol=')[1] for path in paths]


def remove_symbol(symbol: str, tick_type: str):
    path = S3_PATH + f"/{tick_type}/symbol={symbol}/"
    s3fs.rm(path, recursive=True)


def show_symbol_storage_used(symbol: str, tick_type: str) -> dict:
    path = S3_PATH + f"/{tick_type}/symbol={symbol}/"
    return s3fs.du(path)


def get_date_df_from_s3(symbol: str, date: str, tick_type: str, columns: list=None) -> pd.DataFrame:
    byte_data = s3fs.cat(S3_PATH + f"/{tick_type}/symbol={symbol}/date={date}/data.feather")
    if columns:
        df = pd.read_feather(BytesIO(byte_data), columns=columns)
    else:
        df = pd.read_feather(BytesIO(byte_data))
    return df


def date_df_to_file(df: pd.DataFrame, symbol:str, date:str, tick_type: str) -> str:
    path = LOCAL_PATH + f"/{tick_type}/symbol={symbol}/date={date}/"
    Path(path).mkdir(parents=True, exist_ok=True)
    df.to_feather(path+'data.feather', version=2)
    return path + 'data.feather'


def put_date_df_to_s3(df: pd.DataFrame, symbol: str, date: str, tick_type: str) -> pd.DataFrame:
    with NamedTemporaryFile(mode='w+b') as tmp_ref1:
        df.to_feather(path=tmp_ref1.name, version=2)
        s3fs.put(tmp_ref1.name, S3_PATH + f"/{tick_type}/symbol={symbol}/date={date}/data.feather")


def put_df_to_s3(df: pd.DataFrame, tick_type: str, other: str) -> pd.DataFrame:
    with NamedTemporaryFile(mode='w+b') as tmp_ref1:
        df.to_feather(path=tmp_ref1.name, version=2)
        s3fs.put(tmp_ref1.name, S3_PATH + f"/{tick_type}/{other}/data.feather")


def get_and_save_date_df(symbol: str, date: str, tick_type: str) -> pd.DataFrame:
    print(symbol, date, 'getting data fron polygon api')
    df = get_date_df(symbol, date, tick_type)
    print(symbol, date, 'putting data to S3/B2')
    put_date_df_to_s3(df, symbol, date, tick_type)
    print(symbol, date, 'saving data to local file')
    path = date_df_to_file(df, symbol, date, tick_type)
    return df


def fetch_date_df(symbol: str, date: str, tick_type: str) -> pd.DataFrame:
    try:
        print(symbol, date, 'trying to get data from local file...')
        df = pd.read_feather(LOCAL_PATH + f"/{tick_type}/symbol={symbol}/date={date}/data.feather")
    except FileNotFoundError:
        try:
            print(symbol, date, 'trying to get data from s3/b2...')
            df = get_date_df_from_s3(symbol, date, tick_type)
        except FileNotFoundError:
            print(symbol, date, 'getting data from polygon API...')
            df = get_date_df(symbol, date, tick_type)
            print(symbol, date, 'saving data to S3/B2...')
            put_date_df_to_s3(df, symbol, date, tick_type)
        finally:
            print(symbol, date, 'saving data to local file')
            path = date_df_to_file(df, symbol, date, tick_type)

    return df
