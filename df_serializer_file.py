from time import time_ns
from io import BytesIO
from pathlib import Path
import pandas as pd
import pyarrow.feather as pf
from prefect.engine.serializers import Serializer

class FeatherSerializer(Serializer):

    def serialize(self, value: pd.DataFrame) -> bytes:
        # transform a Python object into bytes        
        tmp_filename = str(time_ns()) + '.feather'
        pf.write_feather(
            df=value,
            dest=tmp_filename,
            version=2,
            compression='zstd', # lz4, zstd
            compression_level=5,
            chunksize=64000
        )
        with open(tmp_filename, 'rb') as in_file:
            df_bytes = in_file.read()
        Path(tmp_filename).unlink()
        return df_bytes

    def deserialize(self, value:bytes) -> pd.DataFrame:
        # recover a Python object from bytes
        df_bytes_io = BytesIO(value)
        df = pd.read_feather(df_bytes_io)
        return df


class ParquetSerializer(Serializer):

    def serialize(self, value: pd.DataFrame) -> bytes:
        # transform a Python object into bytes
        tmp_filename = str(time_ns()) + '.parquet'
        value.to_parquet(
            path=tmp_filename,
            index=False
        )
        with open(tmp_filename, 'rb') as in_file:
            df_bytes = in_file.read()
        Path(tmp_filename).unlink()
        return df_bytes

    def deserialize(self, value:bytes) -> pd.DataFrame:
        # recover a Python object from bytes        
        df_bytes_io = BytesIO(value)
        df = pd.read_parquet(df_bytes_io)
        return df
