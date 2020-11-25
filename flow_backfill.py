from datetime import timedelta, date
from psutil import cpu_count
from prefect import Flow, Parameter, task, unmapped
from prefect.engine.executors import DaskExecutor, LocalExecutor
from dates import get_open_market_dates, find_remaining_dates
from polygon_s3 import list_symbol_dates, get_and_save_date_df

 
@task(max_retries=2, retry_delay=timedelta(seconds=2))
def get_remaining_symbol_dates(start_date: str, end_date: str, symbols: list, tick_type: str) -> list:
    request_dates = get_open_market_dates(start_date, end_date)
    symbol_dates = []
    for symbol in symbols:
        existing_dates = list_symbol_dates(symbol, tick_type)
        remaining_dates = find_remaining_dates(request_dates, existing_dates)
        for date in remaining_dates:
            symbol_dates.append((symbol, date))
    return symbol_dates


@task(max_retries=2, retry_delay=timedelta(seconds=2))
def backfill_date_task(symbol_date:tuple, tick_type:str):
    df = get_and_save_date_df(
        symbol=symbol_date[0], 
        date=symbol_date[1], 
        tick_type=tick_type
        )
    return True


def get_flow():
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
    return flow


def run_backfill(symbols: list, tick_type: str, start_date: str, end_date: str=(date.today() - timedelta(days=1)).isoformat(),
    n_workers: int=4, threads_per_worker: int=8, processes: bool=False):

    flow = get_flow()
    # executor = LocalExecutor()
    executor = DaskExecutor(
        cluster_kwargs={
            'n_workers': n_workers,
            'processes': processes,
            'threads_per_worker': threads_per_worker,
        }
    )
    flow_state = flow.run(
        executor=executor,
        symbols=symbols,
        tick_type=tick_type,
        start_date=start_date,
        end_date=end_date,
    )
    return flow_state


if __name__ == '__main__':

    flow_state = run_backfill(
        symbols=['GLD', 'GOLD'], 
        tick_type='trades', 
        start_date='2020-01-01'
        # symbols=['market'], 
        # tick_type='daily', 
        # start_date='2020-01-01'
        )
