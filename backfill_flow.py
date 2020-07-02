from pathlib import Path
from prefect import Flow, task, unmapped
from prefect.engine.results import LocalResult, GCSResult
from prefect.engine.executors import DaskExecutor, LocalDaskExecutor, LocalExecutor
from polygon_rest_api import get_ticker_details
import pandas as pd
from polygon_backfill import backfill_date


# setup results handler
result_filename = "{task_full_name}.prefect"


@task(checkpoint=True, target=result_filename)
def backfill_task(symbol, date)
    backfill_date(symbol, date, result_path, date_partition, tick_type, formats=['feather'])


# result_store = GCSResult(bucket="instasize_prefect_workflows", location=result_filename)
# result_store = LocalResult(dir=Path(__file__).parent.absolute() / 'results', location=result_filename)
result_store = LocalResult(dir='/Users/bobcolner/QuantClarity/tmp', location=result_filename)

with Flow(name="backfill-flow", result=result_store) as flow:
    backfill_task.map()

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
