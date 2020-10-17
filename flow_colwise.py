from pathlib import Path
from prefect import Flow, task, unmapped
from prefect.engine.results import LocalResult, GCSResult
from prefect.engine.executors import DaskExecutor, LocalDaskExecutor, LocalExecutor


# gcloud settings
project_id = "emerald-skill-201716"

# setup results handler
result_filename = "{task_full_name}.prefect"

@task(
    checkpoint=True,
    target=result_filename,
)
def colwise_partial_distcorr_task(df, col1:str, partial:str):
    print(col1)
    df = colwise_partial_distcorr(df, col1, partial)
    return df
    
    
# setup prefect results handeling backend
# result_store = GCSResult(bucket="instasize_prefect_workflows", location=result_filename)
# result_store = LocalResult(dir=Path(__file__).parent.absolute() / 'results', location=result_filename)
result_store = LocalResult(dir='/Users/bobcolner/QuantClarity/tmp', location=result_filename)


# df = npdf
df = npdf.iloc[:, 0:99]

with Flow(name="colwise-flow", result=result_store) as flow:    
    result = colwise_partial_distcorr_task.map(df=unmapped(df), col1=df.columns, partial=unmapped('AAT'))


# executor = LocalExecutor()
# executor = LocalDaskExecutor(scheduler='processes')
executor = DaskExecutor(
    cluster_kwargs={
        'n_workers':4,
        'processes':True,
        'threads_per_worker':1
    }
)
flow.run(executor=executor)
