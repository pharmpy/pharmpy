from pathlib import Path
from typing import Union

from pharmpy.modeling import add_iiv, add_placebo_model, create_basic_pd_model, set_name
from pharmpy.tools.modelfit import create_fit_workflow
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

    fitbase = create_fit_workflow(n=1)
    wb.insert_workflow(fitbase, predecessors=[start_task])

    placebo_task = Task('create_placebo_models', create_placebo_models)
    wb.add_task(placebo_task, predecessors=wb.output_tasks)

    fitplacebo = create_fit_workflow(n=1)
    wb.insert_workflow(fitplacebo, predecessors=wb.output_tasks)

    postprocess_task = Task('postprocess', postprocess)
    wb.add_task(postprocess_task, predecessors=wb.output_tasks)

    return Workflow(wb)


def start_pdsearch(context, dataset):
    context.log_info("Starting pdsearch")

    model = create_basic_pd_model(dataset)
    model = add_iiv(model, ["B"], "exp")
    me = ModelEntry.create(model=model)
    return me


def create_placebo_models(baseme):
    base_model = baseme.model
    linear = add_placebo_model(base_model, "linear")
    linear = set_name(linear, "placebo_linear")
    linear_me = ModelEntry.create(model=linear)
    return linear_me


def postprocess(context, me):
    res = calculate_results()

    context.log_info("Finishing pdsearch")
    return res
