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
        with LocalCluster(processes=False) as cluster:
            with Client(cluster) as client:
                print(client)
                res = client.get(workflow, 'results')
    return res
