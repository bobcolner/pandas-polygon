from os import environ
from io import BytesIO
from tempfile import NamedTemporaryFile
import pandas as pd
from fsspec import filesystem
from s3fs import S3FileSystem
from polygon_backfill import backfill_date_todf


def get_s3fs_client(cached: bool):
    if cached:
        # https://filesystem-spec.readthedocs.io/
        s3 = filesystem(
            protocol='filecache',
            target_protocol='s3',
            target_options={
                'key': environ['B2_ACCESS_KEY_ID'],
                'secret': environ['B2_SECRET_ACCESS_KEY'],
                'client_kwargs': {'endpoint_url': environ['B2_ENDPOINT_URL']}
                },
            cache_storage='/Users/bobcolner/QuantClarity/pandas-polygon/data/cache'
            )
    else:
        s3 = S3FileSystem(
                key=environ['B2_ACCESS_KEY_ID'], 
                secret=environ['B2_SECRET_ACCESS_KEY'], 
                client_kwargs={'endpoint_url': environ['B2_ENDPOINT_URL']}
            )
    return s3


s3 = get_s3fs_client(cached=False)


def list_symbol(symbol:str, tick_type:str='trades') -> str:
    return s3.ls(path=f"polygon-equities/data/{tick_type}/symbol={symbol}/", refresh=True)


def backfill_date_tos3(symbol:str, date:str, tick_type:str) -> pd.DataFrame:
    df = backfill_date_todf(symbol, date, tick_type)
    with NamedTemporaryFile(mode='w+b') as tmp_ref1:
        df.to_feather(path=tmp_ref1.name, version=2)
        s3.put(tmp_ref1.name, f"polygon-equities/data/{tick_type}/symbol={symbol}/date={date}/data.feather")
    return df


def get_tick_df(symbol:str, date:str, tick_type:str='trades', columns=None) -> pd.DataFrame:    
    byte_data = s3.cat(f"polygon-equities/data/{tick_type}/symbol={symbol}/date={date}/data.feather")
    if columns:
        df = pd.read_feather(BytesIO(byte_data), columns=columns)
    else:
        df = pd.read_feather(BytesIO(byte_data))
    return df
