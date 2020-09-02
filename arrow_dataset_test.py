from os import environ
import pandas as pd
import pyarrow as pa
from pyarrow import fs
import pyarrow.dataset as ds


s3  = fs.S3FileSystem(
    access_key=environ['B2_ACCESS_KEY_ID'],
    secret_key=environ['B2_SECRET_ACCESS_KEY'],
    endpoint_override=environ['B2_ENDPOINT_URL']
)

dataset = ds.dataset(
    source='polygon-equities-data',
    format='feather',
    filesystem=s3,
    partitioning='hive',
    # partition_base_dir='feather',
)


df = dataset.to_table(
    # columns=['symbol', 'sip_epoch', 'price', 'size'],
    filter=ds.field('date') == '2020-07-01'
).to_pandas()


# local
dataset = ds.dataset(
    source='/Users/bobcolner/QuantClarity/data/trades/feather/',
    format='feather',
    partitioning='hive',
    # partition_base_dir='trades',
)

from io import BytesIO
bytes_buffer = BytesIO()
fdf.to_parquet(bytes_buffer, index=False)
fdf2 = pd.read_parquet(bytes_buffer)

