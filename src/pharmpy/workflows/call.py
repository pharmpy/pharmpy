from typing import TypeVar

from .context import insert_context
from .workflow import Workflow

T = TypeVar('T')


def call_workflow(wf: Workflow[T], unique_name, db) -> T:
    """Dynamically call a workflow from another workflow.

    Currently only supports dask distributed

    Parameters
    ----------
    wf : Workflow
        A workflow object
    unique_name : str
        A name of the results node that is unique between parent and dynamically created workflows
    db : ToolDatabase
        ToolDatabase to pass to new workflow

    Returns
    -------
    Any
        Whatever the dynamic workflow returns
    """
    from dask.distributed import get_client, rejoin, secede

    from .optimize import optimize_task_graph_for_dask_distributed

    insert_context(wf, db)

    client = get_client()
    dsk = wf.as_dask_dict()
    dsk[unique_name] = dsk.pop('results')
    dsk_optimized = optimize_task_graph_for_dask_distributed(client, dsk)
    futures = client.get(dsk_optimized, unique_name, sync=False)
    secede()
    res: T = client.gather(futures)  # pyright: ignore [reportGeneralTypeIssues]
    rejoin()
    return res
