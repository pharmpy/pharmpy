from pathlib import Path
from typing import Literal, Optional, Union

from pharmpy.modeling import (
    add_iiv,
    add_placebo_model,
    create_basic_pd_model,
    set_description,
    set_direct_effect,
    set_name,
    set_proportional_error_model,
)
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import run_subtool
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder

from .results import calculate_results


def create_workflow(
    dataset: Union[Path, str],
    strictness: str = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    parameter_uncertainty_method: Optional[Literal['SANDWICH', 'SMAT', 'RMAT', 'EFIM']] = None,
):
    """
    Build a PD model

    Parameters
    ----------
    dataset : Union[Path, str]
        A PD dataset
    strictness : str
        Strictness criteria
    parameter_uncertainty_method : {'SANDWICH', 'SMAT', 'RMAT', 'EFIM'} or None
        Parameter uncertainty method. Will be used in ranking models if strictness includes
        parameter uncertainty

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

    placebo_task = Task(
        'run_placebo_models', run_placebo_models, strictness, parameter_uncertainty_method
    )
    wb.add_task(placebo_task, predecessors=base_output)

    de_task = Task(
        'run_drug_effect_models', run_drug_effect_models, strictness, parameter_uncertainty_method
    )
    wb.add_task(de_task, predecessors=wb.output_tasks)

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


def run_placebo_models(context, strictness, parameter_uncertainty_method, baseme):
    exprs = (
        ("linear", "*"),
        ("linear", "+"),
        ("exp", "*"),
        ("hyperbolic", "*"),
        ("hyperbolic", "+"),
    )
    context.log_info(f"Running {len(exprs)} placebo/disease progression models.")
    wb = WorkflowBuilder()
    for expr, op in exprs:
        create_task = Task(f'create_placebo_{expr}_{op}', create_placebo_model, expr, op, baseme)
        wb.add_task(create_task)
        fit_wf = create_fit_workflow(n=1)
        wb.insert_workflow(fit_wf, [create_task])

    def gather(*mes):
        return mes

    gather_task = Task('gather', gather)
    wb.add_task(gather_task, predecessors=wb.output_tasks)

    mes = context.call_workflow(Workflow(wb), "fit-placebo")
    rank_res = run_subtool(
        tool_name='modelrank',
        ctx=context,
        models=[me.model for me in mes] + [baseme.model],
        results=[me.modelfit_results for me in mes] + [baseme.modelfit_results],
        ref_model=baseme.model,
        rank_type='bic_mixed',
        strictness=strictness,
        parameter_uncertainty_method=parameter_uncertainty_method,
        exclude_reference_model=True,
    )

    final_model = rank_res.final_model
    if final_model is None:
        context.abort_workflow("No placebo/disease progression model selected")

    final_me = ModelEntry.create(
        model=rank_res.final_model, modelfit_results=rank_res.final_results
    )

    return final_me


def run_drug_effect_models(context, strictness, parameter_uncertainty_method, baseme):
    exprs = ("linear", "step")
    context.log_info(f"Running {len(exprs)} drug_effect models.")

    wb = WorkflowBuilder()
    for expr in exprs:
        create_task = Task(f'create_drug_effect_{expr}', create_drug_effect_model, expr, baseme)
        wb.add_task(create_task)
        fit_wf = create_fit_workflow(n=1)
        wb.insert_workflow(fit_wf, [create_task])

    def gather(*mes):
        return mes

    gather_task = Task('gather_de', gather)
    wb.add_task(gather_task, predecessors=wb.output_tasks)

    mes = context.call_workflow(Workflow(wb), "fit-drug_effect")
    rank_res = run_subtool(
        tool_name='modelrank',
        ctx=context,
        models=[me.model for me in mes] + [baseme.model],
        results=[me.modelfit_results for me in mes] + [baseme.modelfit_results],
        ref_model=baseme.model,
        rank_type='bic_mixed',
        strictness=strictness,
        parameter_uncertainty_method=parameter_uncertainty_method,
        exclude_reference_model=True,
    )

    final_model = rank_res.final_model
    if final_model is None:
        context.abort_workflow("No drug effect model selected")

    final_me = ModelEntry.create(
        model=rank_res.final_model, modelfit_results=rank_res.final_results
    )

    return final_me


def create_placebo_model(expr, op, baseme):
    base_model = baseme.model
    model = add_placebo_model(base_model, expr, op)
    if op == '*':
        txtop = 'mult'
    elif op == '+':
        txtop = 'add'
    else:
        txtop = op

    model = set_name(model, f"placebo_{expr}_{txtop}")
    model = set_description(model, f"PLACEBO {expr.upper()} {txtop}")
    if expr == 'linear':
        model = add_iiv(model, 'SLOPE', 'prop')
    me = ModelEntry.create(model=model, parent=base_model)
    return me


def create_drug_effect_model(expr, baseme):
    base_model = baseme.model
    model = set_direct_effect(base_model, expr, variable="TIME")
    model = set_name(model, f"drug_{expr}")
    model = set_description(model, f"DRUG {expr.upper()}")
    me = ModelEntry.create(model=model, parent=base_model)
    return me


def postprocess(context, *mes):
    res = calculate_results()

    context.log_info("Finishing pdsearch")
    return res
