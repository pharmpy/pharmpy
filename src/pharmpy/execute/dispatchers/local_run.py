from dask.threaded import get


def run(workflow):
    res = get(workflow, 'results')
    return res
