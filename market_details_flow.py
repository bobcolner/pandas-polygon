from pathlib import Path
import pandas as pd
from prefect import Flow, task, unmapped
from prefect.engine.results import LocalResult, GCSResult
from prefect.engine.executors import DaskExecutor
from polygon_rest_api import get_ticker_details


# setup results handler
result_filename = "{task_full_name}.prefect"


@task(checkpoint=True, target=result_filename)
def symbol_details_task(symbol: str) -> dict:
    print(symbol)
    details = get_ticker_details(symbol)
    return details


@task(checkpoint=True, target=result_filename)
def reduce_list(dict_list:list) -> pd.DataFrame:
    # remove None list elements
    dict_list_nona = [i for i in dict_list if i] 
    return pd.DataFrame(dict_list_nona)


result_store = LocalResult(dir='/Users/bobcolner/QuantClarity/pandas-polygon/data', location=result_filename)

def get_flow():
    with Flow(name="symbol-details-flow", result=result_store) as flow:
        symbols = Parameter('symbols', default=['GLD','SPY'])
        details_list = symbol_details_task.map(symbol=symbols)
        details_df = reduce_list(details_list)
    return flow


def run_flow(symbols: list, n_workers: int=4, threads_per_worker: int=8, processes: bool=False):
    
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
        )
    return flow_state


if __name__ == '__main__':

    flow_state = run_flow(symbols=['GLD', 'GOLD'])

    # get checkpointed data
    out_list = flow_state.result[details_list].result
    out_df = flow_state.result[details_df].result
