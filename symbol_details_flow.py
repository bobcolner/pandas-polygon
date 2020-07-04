from pathlib import Path
from prefect import Flow, task, unmapped
from prefect.engine.results import LocalResult, GCSResult
from prefect.engine.executors import DaskExecutor, LocalDaskExecutor, LocalExecutor
from polygon_rest_api import get_ticker_details
import pandas as pd

# import data
npdf = pd.read_parquet('npdf.parquet')

# setup results handler
result_filename = "{task_full_name}.prefect"


@task(checkpoint=True, target=result_filename)
def symbol_details_task(symbol:str) -> dict:
    print(symbol)
    details = get_ticker_details(symbol)
    return details


@task(checkpoint=True, target=result_filename)
def reduce_list(dict_list:list) -> pd.DataFrame:
    # remove None list elements
    dict_list_nona = [i for i in dict_list if i] 
    return pd.DataFrame(dict_list_nona)


# result_store = GCSResult(bucket="instasize_prefect_workflows", location=result_filename)
# result_store = LocalResult(dir=Path(__file__).parent.absolute() / 'results', location=result_filename)
result_store = LocalResult(dir='/Users/bobcolner/QuantClarity/tmp', location=result_filename)

with Flow(name="symbol-details-flow", result=result_store) as flow:
    
    details_list = symbol_details_task.map(symbol=npdf.columns)
    
    details_df = reduce_list(details_list)


# executor = LocalDaskExecutor(scheduler='processes')
executor = DaskExecutor(
    cluster_kwargs={
        'n_workers':4,
        'processes':True,
        'threads_per_worker':8
    }
)

flow_state = flow.run(executor=executor)

out_list = flow_state.result[details_list].result
out_df = flow_state.result[details_df].result
