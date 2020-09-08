import os
from io import BytesIO
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import s3fs
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


def get_s3_df(symbol:str, date:str, columns=None) -> pd.DataFrame:
    byte_data = s3.cat(f"polygon-equities/data/trades/symbol={symbol}/date={date}/data.feather")
    df = pd.read_feather(BytesIO(byte_data), columns=columns)
    return df
