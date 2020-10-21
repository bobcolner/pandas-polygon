from datetime import timedelta, date
from psutil import cpu_count
import pandas as pd
from prefect import Flow, Parameter, task, unmapped
from prefect.engine.executors import DaskExecutor, LocalExecutor
from polygon_backfill import get_open_market_dates, find_remaining_dates, backfill_date
from polygon_s3 import get_symbol_dates


@task(max_retries=2, retry_delay=timedelta(seconds=2))
def get_remaining_symbol_dates(start_date: str, end_date: str, symbols: list, tick_type: str) -> list:
    request_dates = get_open_market_dates(start_date, end_date)
    symbol_dates = []
    for symbol in symbols:
        existing_dates = get_symbol_dates(symbol, tick_type)
        remaining_dates = find_remaining_dates(request_dates, existing_dates)
        for date in remaining_dates:
            symbol_dates.append((symbol, date))
    return symbol_dates


@task(max_retries=2, retry_delay=timedelta(seconds=2))
def backfill_date_task(symbol_date:tuple, tick_type:str):
    df = backfill_date(
        symbol=symbol_date[0],
        date=symbol_date[1],
        tick_type=tick_type,
        result_path='/Users/bobcolner/QuantClarity/pandas-polygon/data',
        upload_to_s3=True,
        save_local=True
    )
    return True


with Flow(name='backfill-flow') as flow:
    
    start_date = Parameter('start_date', default='2020-01-01')
    end_date = Parameter('end_date', default='2020-02-01')
    tick_type = Parameter('tick_type', default='trades')
    symbols = Parameter('symbols', default=['GLD'])

    symbol_date_list = get_remaining_symbol_dates(start_date, end_date, symbols, tick_type)

    backfill_date_task_result = backfill_date_task.map(
        symbol_date=symbol_date_list,
        tick_type=unmapped(tick_type)
    )


# executor = LocalExecutor()
executor = DaskExecutor(
    cluster_kwargs={
        'n_workers': 2,
        'processes': False,
        'threads_per_worker': 1
    }
)


if __name__ == '__main__':

    flow_state = flow.run(
        executor=executor,
        symbols=['GOLD', 'FSM', 'SPY', 'GLD', 'GDX'],
        # symbols=['market'],
        tick_type='trades',
        start_date='2020-01-01',
        end_date=(date.today() - timedelta(days=1)).isoformat(), # yesterday
    )
