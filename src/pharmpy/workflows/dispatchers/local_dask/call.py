from typing import TypeVar

from ...workflow import Workflow, WorkflowBuilder, insert_context

T = TypeVar('T')


def call_workflow(wf: Workflow[T], unique_name, ctx) -> T:
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

    from .optimize import optimize_task_graph_for_dask_distributed

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


def abort_workflow():
    from dask.distributed import get_client

    client = get_client()
    _turn_of_dask_logging()
    client.close()


def _turn_of_dask_logging():
    # This avoids "WARNING Async instruction for" messages
    # from distributed when aborting the workflow
    import logging

    dask_logger = logging.getLogger("distributed.worker.state_machine")

    def log_filter(record):
        return False

    dask_logger.addFilter(log_filter)
