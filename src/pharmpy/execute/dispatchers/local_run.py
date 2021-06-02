from dask.distributed import Client


def run(workflow):
    with Client(threads_per_worker=1) as client:
        res = client.get(workflow, 'results')
    return res
