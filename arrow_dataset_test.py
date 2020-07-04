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
    format='parquet',
    filesystem=s3,
    partitioning='hive',
    # partition_base_dir='trades',
)


ticks_tbl = dataset.to_table(
    columns=['symbol', 'sip_epoch', 'price', 'size'],
    filter=ds.field('date') == '2020-07-01'
)  # copy from b2

df = ticks_tbl.to_pandas()
