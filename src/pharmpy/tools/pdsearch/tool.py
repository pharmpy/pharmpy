import math
from pathlib import Path
from typing import Literal, Optional, Union

from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import (
    add_iiv,
    add_placebo_model,
    convert_model,
    create_basic_kpd_model,
    create_basic_pd_model,
    get_observations,
    set_description,
    set_direct_effect,
    set_initial_estimates,
    set_name,
    set_proportional_error_model,
)
from pharmpy.tools.common import (
    create_plots,
    table_final_eta_shrinkage,
)
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import run_subtool, summarize_modelfit_results
from pharmpy.workflows import ModelEntry, ModelfitResults, Task, Workflow, WorkflowBuilder

from .results import PDSearchResults


def create_workflow(
    input: Union[Path, str, Model],
    type: Literal['pd', 'kpd'],
    treatment_variable: Optional[str] = None,
    kpd_driver: Literal['ir', 'amount'] = 'ir',
    results: Optional[ModelfitResults] = None,
    strictness: str = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    parameter_uncertainty_method: Optional[Literal['SANDWICH', 'SMAT', 'RMAT', 'EFIM']] = None,
):
    """
    Build a PD model

    Parameters
    ----------
    input : Union[Path, str, Model]
        A PD/KPD dataset or PD/KPD model
    type : str
        Type of PD model to build ('pd' or 'kpd')
    treatment_variable : str
        Name of the variable representing the treatment, e.g. TRT, DOSE or AUC. Do not use if `type` is 'kpd'
    kpd_driver : str
        Driver for KPD model (virtual infusion rate 'ir' or 'amount')
    results : ModelfitResults (optional)
        Results to input model
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

    start_task = Task('start_pdsearch', start_pdsearch, input, type, kpd_driver, results)
    wb.add_task(start_task)

    if isinstance(input, Model):
        base_output = [start_task]
    else:
        fitbase = create_fit_workflow(n=1)
        wb.insert_workflow(fitbase, predecessors=[start_task])
        base_output = wb.output_tasks

    placebo_task = Task(
        'run_placebo_models', run_placebo_models, strictness, parameter_uncertainty_method
    )
    wb.add_task(placebo_task, predecessors=base_output)

    de_task = Task(
        'run_drug_effect_models',
        run_drug_effect_models,
        treatment_variable,
        strictness,
        parameter_uncertainty_method,
    )
    wb.add_task(de_task)

    postprocess_task = Task('postprocess', postprocess)
    wb.add_task(postprocess_task, predecessors=de_task)

    wb.scatter(placebo_task, (de_task, postprocess_task))

    return Workflow(wb)


def _calc_pd_inits_from_data(model):
    idc = model.datainfo.id_column.name
    obs = get_observations(model)
    theta_init = obs.groupby(idc).median().median()
    variance = (obs.groupby(idc).std() ** 2).mean()
    mean = obs.groupby(idc).mean().mean()
    omega_init = math.log((1 + variance / mean**2))
    return theta_init, omega_init


def start_pdsearch(context, input, type, kpd_driver, results):
    context.log_info("Starting pdsearch")
    if isinstance(input, Model):
        me = ModelEntry.create(input, modelfit_results=results)
        context.store_input_model_entry(me)
        return me
    else:
        model = create_base_model(type, input, kpd_driver)
        me = ModelEntry.create(model=model)
        return me


def create_base_model(type, dataset, kpd_driver):
    if type == 'pd':
        model = create_basic_pd_model(dataset)
    else:
        model = create_basic_kpd_model(dataset, driver=kpd_driver)
    model = set_proportional_error_model(model, zero_protection=False)
    model = add_iiv(model, ["B"], "exp")
    if isinstance(dataset, Path):
        theta_init, omega_init = _calc_pd_inits_from_data(model)
        model = set_initial_estimates(model, {"POP_B": theta_init, "IIV_B": omega_init})
    model = convert_model(model, 'nonmem')  # FIXME: Workaround for results retrieval system
    return model


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

    wb.gather(wb.output_tasks)

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

    return final_me, rank_res


def run_drug_effect_models(
    context, treatment_variable, strictness, parameter_uncertainty_method, baseme
):
    exprs = ("linear", "step", "emax", "sigmoid")
    context.log_info(f"Running {len(exprs)} drug_effect models.")

    wb = WorkflowBuilder()
    for expr in exprs:
        create_task = Task(
            f'create_drug_effect_{expr}', create_drug_effect_model, treatment_variable, expr, baseme
        )
        wb.add_task(create_task)
        fit_wf = create_fit_workflow(n=1)
        wb.insert_workflow(fit_wf, [create_task])

    wb.gather(wb.output_tasks)

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

    return rank_res


def create_placebo_model(expr, op, baseme):
    base_model = baseme.model
    model = set_initial_estimates(base_model, baseme.modelfit_results.parameter_estimates)
    model = add_placebo_model(model, expr, op)
    if op == '*':
        txtop = 'mult'
    elif op == '+':
        txtop = 'add'
    else:
        txtop = op

    model = set_name(model, f"placebo_{expr}_{txtop}")
    model = set_description(model, f"PLACEBO({expr.upper()} {txtop})")
    if expr == 'linear':
        model = add_iiv(model, 'SLOPE', 'prop')
    me = ModelEntry.create(model=model, parent=base_model)
    return me


def create_drug_effect_model(treatment_variable, expr, baseme):
    if not treatment_variable:
        treatment_variable = 'KPD'
    base_model = baseme.model
    model = set_initial_estimates(base_model, baseme.modelfit_results.parameter_estimates)
    model = set_direct_effect(model, expr, variable=treatment_variable)
    model = set_name(model, f"drug_{expr}")
    model = set_description(model, model.description + f"; DIRECTEFFECT({expr.upper()})")
    me = ModelEntry.create(model=model, parent=base_model)
    return me


def postprocess(context, rank_res, rank_res2):
    step1 = rank_res.summary_tool.assign(step=1)
    step2 = rank_res2.summary_tool.assign(step=2)
    summary_tool = pd.concat((step1, step2))
    summary_tool = summary_tool.reset_index().set_index(["model", "step"])
    summary_models = summarize_modelfit_results(context)

    plots = create_plots(rank_res2.final_model, rank_res2.final_results)
    eta_shrinkage = table_final_eta_shrinkage(rank_res2.final_model, rank_res2.final_results)

    res = PDSearchResults(
        summary_tool=summary_tool,
        summary_models=summary_models,
        final_model=rank_res2.final_model,
        final_results=rank_res2.final_results,
        final_model_dv_vs_ipred_plot=plots['dv_vs_ipred'],
        final_model_dv_vs_pred_plot=plots['dv_vs_pred'],
        final_model_cwres_vs_idv_plot=plots['cwres_vs_idv'],
        final_model_abs_cwres_vs_ipred_plot=plots['abs_cwres_vs_ipred'],
        final_model_eta_distribution_plot=plots['eta_distribution'],
        final_model_eta_shrinkage=eta_shrinkage,
    )

    context.log_info("Finishing pdsearch")
    return res


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    input,
    type,
    treatment_variable,
    kpd_driver,
    results,
    strictness,
    parameter_uncertainty_method,
):
    if type == 'pd' and treatment_variable is None:
        raise ValueError('Invalid `treatment_variable`: must be specified when type is `pd`')
