from pathlib import Path
from typing import Union

from pharmpy.modeling import create_basic_pd_model
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder

from .results import calculate_results


def create_workflow(
    dataset: Union[Path, str],
):
    """
    Build a PD model

    Parameters
    ----------
    dataset : Union[Path, str]
        A PD dataset

    Returns
    -------
    PDSearchResults
        PDSearch tool results object.

    """
    wb = WorkflowBuilder(name="pdsearch")

    start_task = Task('start_pdsearch', start_pdsearch, dataset)
    wb.add_task(start_task)
    postprocess_task = Task('postprocess', postprocess)
    wb.add_task(postprocess_task, predecessors=[start_task])

    return Workflow(wb)


def start_pdsearch(context, dataset):
    context.log_info("Starting pdsearch")

    model = create_basic_pd_model(dataset)
    me = ModelEntry.create(model=model)
    return me


def postprocess(context, me):
    res = calculate_results()

    context.log_info("Finishing pdsearch")
    return res
