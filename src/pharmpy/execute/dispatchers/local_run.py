from dask.diagnostics import ProgressBar
from dask.threaded import get


def run(workflow):
    with ProgressBar(dt=1):
        res = get(workflow, 'results')
    return res
