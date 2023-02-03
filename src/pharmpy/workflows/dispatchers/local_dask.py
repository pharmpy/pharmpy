import os
import warnings
from typing import TypeVar

import pharmpy.workflows.dispatchers
from pharmpy.internals.fs.cwd import chdir
from pharmpy.internals.fs.tmp import TemporaryDirectory

from ..workflow import Workflow

T = TypeVar('T')


def run(workflow: Workflow[T]) -> T:
    # NOTE We change to a new temporary directory so that all files generated
    # by the workflow end-up in the same root directory. Each task of a
    # workflow has the responsibility to avoid collisions on the file system
    # (using UUIDs as filenames for instance). This also allows tasks to share
    # files on the local file-system if needed. Since in the future we may want
    # to run tasks on remote machines it might be a good idea to instead not
    # assume tasks share the same working directory or even filesystem, and
    # let them communicate through some other means and create their own
    # temporary directory if needed. That being said, tasks should certainly
    # not change their cwd because they could be running on threads and sharing
    # the cwd with others.
    with TemporaryDirectory() as tempdirname, chdir(tempdirname):
        dsk = workflow.as_dask_dict()

        if pharmpy.workflows.dispatchers.conf.dask_dispatcher:
            dask_dispatcher = pharmpy.workflows.dispatchers.conf.dask_dispatcher
        else:
            dask_dispatcher = 'distributed'

        if dask_dispatcher == 'threaded':
            from dask.threaded import get

            res = get(dsk, 'results')
        else:
            import dask
            from dask.distributed import LocalCluster  # pyright: ignore [reportPrivateImportUsage]
            from dask.distributed import Client

            from ..optimize import optimize_task_graph_for_dask_distributed

            # NOTE We set the dask temporary directory to avoid permission
            # errors in the dask-worker-space directory in case for
            # instance different users run dask on the same filesystem (e.g. on
            # a cluster node).
            # An attempt at solving this kind of problems was introduced two
            # months ago, by suffixing the directory name with the user id on
            # POSIX. It should fix the problem, except maybe on Windows. Since
            # we experienced issues before that change, maybe this is it and we
            # could instead set temporary_directory to tempfile.gettempdir().
            # Note that if this is done, then our custom patch of
            # TemporaryDirectory can gotten rid of. See:
            #   - https://github.com/dask/distributed/blob/cff33d500f24b67efbd94ce39b15cb36473cd9f6/distributed/diskutils.py#L132-L153 # noqa: E501
            #   - https://github.com/dask/distributed/issues/6748
            #   - https://github.com/dask/distributed/pull/7054
            # NOTE We also ignore cleanup errors that can occur on Windows. We
            # must do so because dask also ignore those, see for instance:
            #   - https://github.com/dask/distributed/issues/6052#issue-1189891052
            #   - https://github.com/dask/distributed/issues/966#issuecomment-353265964
            #   - https://github.com/dask/distributed/commit/9ffac1b9b
            #   - https://github.com/dask/distributed/blob/5dc591bbdd4427fe49fe90338a34fc85ee35f2c9/distributed/diskutils.py#L23-L29  # noqa: E501
            #   - https://github.com/dask/distributed/commit/7ed517c47de90a68abd537d29df9740a2c20b638
            is_windows = os.name == 'nt'
            with TemporaryDirectory(
                ignore_cleanup_errors=is_windows
            ) as dasktempdir, dask.config.set(  # pyright: ignore [reportPrivateImportUsage]
                {'temporary_directory': dasktempdir}
            ):
                with warnings.catch_warnings():
                    # NOTE Catch deprecation warning from python 3.10 via tornado.
                    # Should be fixed with tornado 6.2
                    warnings.filterwarnings("ignore", message="There is no current event loop")
                    with LocalCluster(
                        processes=False, dashboard_address=':31058'
                    ) as cluster, Client(cluster) as client:
                        print(client)
                        dsk_optimized = optimize_task_graph_for_dask_distributed(client, dsk)
                        res = client.get(dsk_optimized, 'results')
    return res  # pyright: ignore [reportGeneralTypeIssues]
