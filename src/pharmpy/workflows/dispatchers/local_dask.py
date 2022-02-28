import warnings

import pharmpy.workflows.dispatchers
from pharmpy.utils import TemporaryDirectory, TemporaryDirectoryChanger


def run(workflow):

    with TemporaryDirectory() as tempdirname:
        with TemporaryDirectoryChanger(tempdirname):
            workflow = workflow.as_dask_dict()

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

                with warnings.catch_warnings():
                    # Catch deprecation warning from python 3.10 via tornado.
                    # Should be fixed with tornado 6.2
                    warnings.filterwarnings("ignore", message="There is no current event loop")
                    with LocalCluster(processes=False) as cluster:
                        with Client(cluster) as client:
                            print(client)
                            res = client.get(workflow, 'results')
    return res
