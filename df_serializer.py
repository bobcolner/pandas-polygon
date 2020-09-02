from io import BytesIO
import pandas as pd
import pyarrow.feather as pf
from prefect.engine.serializers import Serializer


class FeatherSerializer(Serializer):

    def serialize(self, value: pd.DataFrame) -> bytes:
        # transform a Python object into bytes        
        bytes_buffer = BytesIO()
        pf.write_feather(
            df=value,
            dest=bytes_buffer,
            version=2,
        )
        return bytes_buffer.getvalue()

    def deserialize(self, value:bytes) -> pd.DataFrame:
        # recover a Python object from bytes
        df_bytes_io = BytesIO(value)
        return pd.read_feather(df_bytes_io)


class ParquetSerializer(Serializer):

    def serialize(self, value: pd.DataFrame) -> bytes:
        # transform a Python object into bytes
        bytes_buffer = BytesIO()
        value.to_parquet(
            path=bytes_buffer,
            index=False
        )
        return bytes_buffer.getvalue()

    def deserialize(self, value:bytes) -> pd.DataFrame:
        # recover a Python object from bytes        
        df_bytes_io = BytesIO(value)
        return pd.read_parquet(df_bytes_io)
