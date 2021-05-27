from dask.diagnostics import ProgressBar
from dask.multiprocessing import get


def run(workflow):
    with ProgressBar(dt=1):
        # FIXME: issue with NONMEM license file, use multiprocessing instead of threads
        #  as temporary solution
        res = get(workflow, 'results')
    return res
