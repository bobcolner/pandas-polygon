from os import environ
from io import BytesIO
from tempfile import NamedTemporaryFile
from pathlib import Path
import pandas as pd
from fsspec import filesystem
from s3fs import S3FileSystem


def get_s3fs_client(cached: bool=False):
    if cached:
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
        s3fs = S3FileSystem(
                key=environ['B2_ACCESS_KEY_ID'], 
                secret=environ['B2_SECRET_ACCESS_KEY'], 
                client_kwargs={'endpoint_url': environ['B2_ENDPOINT_URL']}
            )
    return s3fs


s3fs = get_s3fs_client(cached=False)


def list_symbol_dates(symbol: str, tick_type: str='trades') -> str:
    return s3fs.ls(path=f"polygon-equities/data/{tick_type}/symbol={symbol}/", refresh=True)


def put_date_to_s3(symbol: str, date: str, tick_type: str) -> pd.DataFrame:
    df = get_ticks_date_df(symbol, date, tick_type)
    with NamedTemporaryFile(mode='w+b') as tmp_ref1:
        df.to_feather(path=tmp_ref1.name, version=2)
        s3fs.put(tmp_ref1.name, f"polygon-equities/data/{tick_type}/symbol={symbol}/date={date}/data.feather")
    return df


def get_date_from_s3(symbol: str, date: str, tick_type: str='trades', columns=None) -> pd.DataFrame:    
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
        df = pd.read_feather(local_path + f"{tick_type}/symbol={symbol}/date={date}/data.feather")
    except FileNotFoundError:
        try:
            print('trying to get ticks from s3...')
            df = get_date_from_s3(symbol, date, tick_type)
        except FileNotFoundError:
            print('trying to get ticks from polygon API and save to s3...')
            df = put_date_to_s3(symbol, date, tick_type)
        finally:
            print('saving ticks to local file')
            path = date_df_to_file(df, symbol, date, tick_type, local_path)
    return df
