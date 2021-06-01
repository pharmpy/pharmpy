# from dask.diagnostics import ProgressBar
from dask.multiprocessing import get

# from dask.distributed import Client


def run(workflow):
    # with Client() as client:
    #    res = client.get(workflow, 'results')
    res = get(workflow, 'results')
    return res
