from datetime import date
from pathlib import Path
from itertools import product
from prefect import Flow, Parameter, task, unmapped
from prefect.engine.results import LocalResult, GCSResult, S3Result
from prefect.engine.executors import DaskExecutor, LocalDaskExecutor, LocalExecutor
from df_serializer import FeatherSerializer, ParquetSerializer
import pandas as pd
from polygon_backfill import backfill_date_todf, get_open_market_dates


@task(checkpoint=False)
def cross_product_task(x, y) -> list:
    return list(product(x, y))


@task(checkpoint=False)
def requested_dates_task(start_date, end_date):
    requested_dates = get_open_market_dates(start_date, end_date)
    print(requested_dates)
    return requested_dates


@task(
    checkpoint=True, 
    target="us_stocks/{tick_type}/symbol={symbol}/date={backfill_date}/data.feather"
)
def backfill_task(symbol:str, backfill_date:str, tick_type:str) -> pd.DataFrame:
    df = backfill_date_todf(symbol=symbol, date=backfill_date, tick_type=tick_type)
    return df


result_store = S3Result(
    serializer=FeatherSerializer(),
    bucket='polygon-equities-data', 
    boto3_kwargs={
        'aws_access_key_id':'000bc68c5651ba80000000001',
        'aws_secret_access_key':'K000ClFsM+jVS18gf0332g7gNHRTWt8',
        'endpoint_url':'https://s3.us-west-000.backblazeb2.com',
    },
)
# result_store = GCSResult(
#     bucket='emerald-skill-datascience', 
#     serializer=FeatherSerializer()
# )
# result_store = LocalResult(
#     dir='/Users/bobcolner/QuantClarity/data', 
#     # dir=Path(__file__).parent.absolute() / 'results'
#     serializer=FeatherSerializer()
# )

symbols = ['SPY','GLUU','IHI','NVDA']
dates = ['2020-06-22', '2020-07-01','2020-07-02']

with Flow(name='backfill-flow', result=result_store) as flow:
    
    start_date = Parameter('start_date', default='2020-01-01')
    end_date = Parameter('end_date', default='2020-02-01')
    symbol = Parameter('symbol', default='SPY')
    tick_type = Parameter('tick_type', default='trades')

    req_dates = requested_dates_task(start_date, end_date)

    # backfill_task_result = backfill_task.map(
    #     cross_product_task(symbols, dates),
    #     tick_type=unmapped('trades')
    # )
    backfill_task_result = backfill_task.map(
        symbol=unmapped(symbol),
        backfill_date=req_dates,
        tick_type=unmapped(tick_type)
    )


executor = LocalDaskExecutor(scheduler='threads')
# executor = LocalExecutor()
# executor = DaskExecutor(
#     cluster_kwargs={
#         'n_workers':4,
#         'processes':True,
#         'threads_per_worker':8
#     }
# )

flow_state = flow.run(
    executor=executor, 
    symbol='SPY', 
    tick_type='trades',
    start_date='2019-01-01', 
    end_date=date.today().isoformat(),
)

# result_obj = flow_state.result[backfill_task_result].result
# result_obj
