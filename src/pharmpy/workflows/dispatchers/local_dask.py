import os
import signal
import warnings
from typing import TypeVar

import pharmpy.workflows.dispatchers
from pharmpy.internals.fs.cwd import chdir
from pharmpy.internals.fs.tmp import TemporaryDirectory

from ..workflow import Workflow, WorkflowBuilder, insert_context
from .baseclass import Dispatcher
from .slurm_helpers import get_slurm_nodename, is_running_on_slurm

T = TypeVar('T')


class LocalDaskDispatcher(Dispatcher):
    def run(self, workflow: Workflow[T], context) -> T:
        # NOTE: We change to a new temporary directory so that all files generated
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
                from dask.distributed import Client, LocalCluster

                # NOTE: We set the dask temporary directory to avoid permission
                # errors in the dask-worker-space directory in case for
                # instance different users run dask on the same filesystem (e.g., on
                # a cluster node).
                # An attempt at solving this kind of problem was introduced two
                # months ago, by suffixing the directory name with the user id on
                # POSIX. It should fix the problem, except maybe on Windows. Since
                # we experienced issues before that change, maybe this is it, and we
                # could instead set temporary_directory to tempfile.gettempdir().
                # Note that if this is done, then our custom patch of
                # TemporaryDirectory can be removed. See:
                #   - https://github.com/dask/distributed/blob/cff33d500f24b67efbd94ce39b15cb36473cd9f6/distributed/diskutils.py#L132-L153 # noqa: E501
                #   - https://github.com/dask/distributed/issues/6748
                #   - https://github.com/dask/distributed/pull/7054
                # NOTE: We also ignore cleanup errors that can occur on Windows. We
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
                        # NOTE: Catch deprecation warning from python 3.10 via tornado.
                        # Should be fixed with tornado 6.2
                        warnings.filterwarnings("ignore", message="There is no current event loop")
                        # Because of https://github.com/dask/distributed/issues/8559 when having no network
                        warnings.filterwarnings(
                            "ignore", "Couldn't detect a suitable IP address for reaching"
                        )
                        # When processes=False the number of workers is 1. Use threads per worker option to parallelize
                        ncores = context.retrieve_dispatching_options()['ncores']
                        dashboard_address = '31058'
                        with LocalCluster(
                            processes=False,
                            dashboard_address=f':{dashboard_address}',
                            threads_per_worker=ncores,
                        ) as cluster, Client(cluster) as client:
                            context.log_info(
                                "Dispatching workflow with local_dask dispatcher "
                                f"in {context}: {client} at dashboard address {dashboard_address}"
                            )
                            dsk_optimized = optimize_task_graph_for_dask_distributed(client, dsk)

                            def sigint_handler(sig, frame):
                                context.abort_workflow("Workflow was interrupted by user (SIGINT)")

                            def sigterm_handler(sig, frame):
                                context.abort_workflow("Workflow was terminated (SIGTERM)")

                            def sighup_handler(sig, frame):
                                # If the calling shell dies but we are still alive
                                # we want to block SIGPIPE in case we are writing
                                # to the now broken stdout. This way we could
                                # still be able to run the workflow to completion
                                signal.signal(signal.SIGPIPE, signal.SIG_IGN)

                            signal.signal(signal.SIGINT, sigint_handler)
                            signal.signal(signal.SIGTERM, sigterm_handler)
                            if os.name != 'nt':
                                # Windows doesn't recognize the SIGHUP signal
                                signal.signal(signal.SIGHUP, sighup_handler)

                            try:
                                res = client.get(dsk_optimized, 'results')
                            except dask.distributed.client.FutureCancelledError:
                                res = None
                            context.log_info("End dispatch")
        return res  # pyright: ignore [reportGeneralTypeIssues]

    def call_workflow(self, wf: Workflow[T], unique_name, ctx) -> T:
        """Dynamically call a workflow from another workflow.

        Currently only supports dask distributed

        Parameters
        ----------
        wf : Workflow
            A workflow object
        unique_name : str
            A name of the results node that is unique between parent and dynamically created workflows
        ctx : Context
            Context to pass to new workflow

        Returns
        -------
        Any
            Whatever the dynamic workflow returns
        """
        from dask.distributed import get_client, rejoin, secede

        wb = WorkflowBuilder(wf)
        insert_context(wb, ctx)
        wf = Workflow(wb)

        client = get_client()
        dsk = wf.as_dask_dict()
        dsk[unique_name] = dsk.pop('results')
        dsk_optimized = optimize_task_graph_for_dask_distributed(client, dsk)
        futures = client.get(dsk_optimized, unique_name, sync=False)
        secede()
        res: T = client.gather(futures)  # pyright: ignore [reportGeneralTypeIssues]
        rejoin()
        return res

    def abort_workflow(self):
        from dask.distributed import get_client

        client = get_client()
        _turn_off_dask_logging()
        client.close()

    def get_hosts(self) -> dict[str, int]:
        hosts = {'localhost': os.cpu_count()}
        return hosts

    def get_hostname(self) -> str:
        if is_running_on_slurm():
            return get_slurm_nodename()
        else:
            return 'localhost'

    def get_available_cores(self, allocation: int):
        return 1


def _turn_off_dask_logging():
    # This avoids "WARNING Async instruction for" messages
    # from distributed when aborting the workflow
    import logging

    dask_logger = logging.getLogger("distributed.worker.state_machine")

    def log_filter(record):
        return False

    dask_logger.addFilter(log_filter)


def optimize_task_graph_for_dask_distributed(client, graph):
    from dask.distributed import Future

    optimized = {key: _scatter_computation(Future, client, value) for key, value in graph.items()}
    from dask.optimization import fuse

    fused = fuse(optimized)[0]
    return fused


def _scatter_computation(Future, client, computation):
    # NOTE: According to dask's graph spec (https://docs.dask.org/en/stable/spec.html):
    # A computation may be one of the following:
    #  - Any key present in the Dask graph like 'x'
    #  - Any other value like 1, to be interpreted literally
    #  - A task like (inc, 'x') (see below)
    #  - A list of computations, like [1, 'x', (inc, 'x')]
    if isinstance(computation, tuple):
        if len(computation) == 0:  # Avoid further interpreting empty argument
            return computation
        else:
            return (
                computation[0],
                *map(lambda c: _scatter_computation(Future, client, c), computation[1:]),
            )

    if isinstance(computation, list):
        return list(map(lambda c: _scatter_computation(Future, client, c), computation))

    return _scatter_value(Future, client, computation)


def _scatter_value(Future, client, value):
    import dask

    # TODO: We could automatically compute whether object size is above
    # threshold with a slight twist on https://stackoverflow.com/a/30316760
    if isinstance(value, (dict, int, str, float, bool, range, Future)) or callable(value):
        return value
    else:
        dask_version = tuple(int(i) for i in dask.__version__.split('.'))
        if dask_version >= (2024, 2, 1):
            # This is a workaround for https://github.com/dask/distributed/issues/8576
            future = client.scatter(value, hash=False)
        else:
            future = client.scatter(value)
        return future
