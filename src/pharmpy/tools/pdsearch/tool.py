from pathlib import Path
from typing import Union

from pharmpy.modeling import (
    add_iiv,
    add_placebo_model,
    create_basic_pd_model,
    set_description,
    set_name,
    set_proportional_error_model,
)
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
    base_output = wb.output_tasks

    placebo_task = Task('run_placebo_models', run_placebo_models)
    wb.add_task(placebo_task, predecessors=base_output)

    postprocess_task = Task('postprocess', postprocess)
    wb.add_task(postprocess_task, predecessors=wb.output_tasks)

    return Workflow(wb)


def start_pdsearch(context, dataset):
    context.log_info("Starting pdsearch")

    model = create_basic_pd_model(dataset)
    model = set_proportional_error_model(model, zero_protection=False)
    model = add_iiv(model, ["B"], "exp")
    me = ModelEntry.create(model=model)
    return me


def run_placebo_models(context, baseme):
    exprs = ("linear", "exp")
    wb = WorkflowBuilder()
    for expr in exprs:
        create_task = Task(f'create_placebo_{expr}', create_placebo_model, expr, baseme)
        wb.add_task(create_task)
        fit_wf = create_fit_workflow(n=1)
        wb.insert_workflow(fit_wf, [create_task])

    def gather(*mes):
        return mes

    gather_task = Task('gather', gather)
    wb.add_task(gather_task, predecessors=wb.output_tasks)

    mes = context.call_workflow(Workflow(wb), "fit-placebo")
    return mes


def create_placebo_model(expr, baseme):
    base_model = baseme.model
    model = add_placebo_model(base_model, expr)
    model = set_name(model, f"placebo_{expr}")
    model = set_description(model, f"PLACEBO {expr.upper()}")
    if expr == 'linear':
        model = add_iiv(model, 'SLOPE', 'prop')
    me = ModelEntry.create(model=model, parent=base_model)
    return me


def postprocess(context, *mes):
    res = calculate_results()

    context.log_info("Finishing pdsearch")
    return res
