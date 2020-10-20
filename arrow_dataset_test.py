from os import environ
import pandas as pd
import pyarrow as pa
from pyarrow import fs
import pyarrow.dataset as ds


def get_s3_dataset():
    s3  = fs.S3FileSystem(
        access_key=environ['B2_ACCESS_KEY_ID'],
        secret_key=environ['B2_SECRET_ACCESS_KEY'],
        endpoint_override=environ['B2_ENDPOINT_URL']
    )

    dataset = ds.dataset(
        source='polygon-equities/data/trades',
        format='feather',
        filesystem=s3,
        partitioning='hive',
        exclude_invalid_files=True
    )
    return dataset


def get_local_dataset():
    dataset = ds.dataset(
        source='/Users/bobcolner/QuantClarity/data/trades/',
        format='feather',
        partitioning='hive',
    )
    return dataset

df = dataset.to_table(
    # columns=['symbol', 'sip_epoch', 'price', 'size'],
    filter=ds.field('date') == '2020-07-01'
).to_pandas()


from io import BytesIO
bytes_buffer = BytesIO()
fdf.to_parquet(bytes_buffer, index=False)
fdf2 = pd.read_parquet(bytes_buffer)

