import pharmpy.workflows.dispatchers


def run(workflow):

    if pharmpy.workflows.dispatchers.conf.dask_dispatcher:
        dask_dispatcher = pharmpy.workflows.dispatchers.conf.dask_dispatcher
    else:
        dask_dispatcher = 'distributed'

    if dask_dispatcher == 'threaded':
        from dask.threaded import get

        res = get(workflow, 'results')
    else:
        import tempfile

        import dask
        from dask.distributed import Client, LocalCluster

        dask.config.set({'temporary_directory': tempfile.gettempdir()})
        import os

        if os.name == 'nt':
            # Parallelization does not work in R on Windows
            # always disable on Windows as workaround.
            processes = False
        else:
            processes = True
        with LocalCluster(threads_per_worker=1, processes=processes) as cluster:
            with Client(cluster) as client:
                res = client.get(workflow, 'results')
    return res
