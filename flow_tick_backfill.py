from os import environ
from datetime import date
from itertools import product
from psutil import cpu_count
from prefect import Flow, Parameter, task, unmapped
from prefect.engine.results import S3Result
from prefect.engine.executors import DaskExecutor, LocalExecutor, LocalDaskExecutor
import pandas as pd
from polygon_backfill import get_open_market_dates, backfill_date


@task(checkpoint=False)
def cross_product_task(x, y) -> list:
    return list(product(x, y))


@task(checkpoint=False)
def requested_dates_task(start_date: str, end_date: str) -> list:
    return get_open_market_dates(start_date, end_date)


@task(
    checkpoint=True,
    target="checkpoints/{tick_type}/symbol={symbol_date[0]}/date={symbol_date[1]}/data.prefect"
)
def backfill_task(symbol_date:tuple, tick_type:str) -> pd.DataFrame:
    df = backfill_date(
        symbol=symbol_date[0],
        date=symbol_date[1],
        tick_type=tick_type,
        result_path='/Users/bobcolner/QuantClarity/pandas-polygon/data', 
        upload_to_s3=True,
        save_local=True
        )
    return True


result_store = S3Result(
    bucket='polygon-equities', 
    boto3_kwargs={
        'aws_access_key_id': environ['B2_ACCESS_KEY_ID'],
        'aws_secret_access_key': environ['B2_SECRET_ACCESS_KEY'],
        'endpoint_url':  environ['B2_ENDPOINT_URL']
    }
)


with Flow(name='backfill-flow', result=result_store) as flow:
    
    start_date = Parameter('start_date', default='2020-01-01')
    end_date = Parameter('end_date', default='2020-02-01')
    tick_type = Parameter('tick_type', default='trades')
    symbols = Parameter('symbols', default=['GLD'])

    request_dates = requested_dates_task(start_date, end_date)

    symbol_date_list = cross_product_task(symbols, request_dates)

    backfill_task_result = backfill_task.map(
        symbol_date=symbol_date_list,
        tick_type=unmapped(tick_type)
    )


# executor = LocalExecutor()
# executor = LocalDaskExecutor(scheduler='threads')
executor = DaskExecutor(
    cluster_kwargs={
        'n_workers': cpu_count(),
        'processes': True,
        'threads_per_worker': 8
    }
)


if __name__ == '__main__':

    flow_state = flow.run(
        executor=executor,
        symbols=['GLD'],
        tick_type='trades',
        start_date='2020-01-01',
        end_date=date.today().isoformat(),
    )
