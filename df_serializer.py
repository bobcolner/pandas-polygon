from tempfile import NamedTemporaryFile
from io import BytesIO
import pandas as pd
import pyarrow.feather as pf
from prefect.engine.serializers import Serializer

class FeatherSerializer(Serializer):

    def serialize(self, value: pd.DataFrame) -> bytes:
        # transform a Python object into bytes        
        with NamedTemporaryFile(mode='w+b') as tmp_ref1, open(tmp_ref1.name, mode='rb') as tmp_ref2:
            # write to tmp file
            pf.write_feather(
                df=value,
                dest=tmp_ref1,
                compression='zstd', # lz4, zstd
                compression_level=5,
                chunksize=64000,
                version=2,
            )
            # read bytes from tmp file
            df_bytes = tmp_ref2.read()
        
        return df_bytes

    def deserialize(self, value:bytes) -> pd.DataFrame:
        # recover a Python object from bytes
        df_bytes_io = BytesIO(value)
        df = pd.read_feather(df_bytes_io)
        return df


class ParquetSerializer(Serializer):

    def serialize(self, value: pd.DataFrame) -> bytes:
        # transform a Python object into bytes
        with NamedTemporaryFile(mode='w+b') as tmp_ref1, open(tmp_ref1.name, 'rb') as tmp_ref2:
            # write to tmp file
            value.to_parquet(
                path=tmp_ref1,
                index=False
            )
            # read bytes from tmp file
            df_bytes = tmp_ref2.read()

        return df_bytes

    def deserialize(self, value:bytes) -> pd.DataFrame:
        # recover a Python object from bytes        
        df_bytes_io = BytesIO(value)
        df = pd.read_parquet(df_bytes_io)
        return df
