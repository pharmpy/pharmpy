from dask.distributed import Client


def run(workflow):
    with Client() as client:
        res = client.get(workflow, 'results')
    return res
