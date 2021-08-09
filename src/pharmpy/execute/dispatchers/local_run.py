import os

from dask.distributed import Client
from dask.threaded import get

import pharmpy.execute.dispatchers


def run(workflow):
    if pharmpy.execute.dispatchers.conf.dask_dispatcher:
        dask_dispatcher = pharmpy.execute.dispatchers.conf.dask_dispatcher
    elif os.name == 'nt':
        dask_dispatcher = 'threaded'
    else:
        dask_dispatcher = 'distributed'

    if dask_dispatcher == 'threaded':
        res = get(workflow, 'results')
    else:
        with Client(threads_per_worker=1) as client:
            res = client.get(workflow, 'results')
    return res
