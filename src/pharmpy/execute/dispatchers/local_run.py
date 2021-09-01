import pharmpy.execute.dispatchers


def run(workflow):
    from dask.distributed import Client
    from dask.threaded import get

    if pharmpy.execute.dispatchers.conf.dask_dispatcher:
        dask_dispatcher = pharmpy.execute.dispatchers.conf.dask_dispatcher
    else:
        dask_dispatcher = 'distributed'

    if dask_dispatcher == 'threaded':
        res = get(workflow, 'results')
    else:
        with Client(threads_per_worker=1, processes=False) as client:
            res = client.get(workflow, 'results')
    return res
