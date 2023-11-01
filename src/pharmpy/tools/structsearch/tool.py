from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.tools import summarize_modelfit_results
from pharmpy.tools.common import ToolResults, create_results, update_initial_estimates
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow, WorkflowBuilder, call_workflow
from pharmpy.workflows.results import ModelfitResults

from .drugmetabolite import create_base_metabolite, create_drug_metabolite_models
from .pkpd import create_baseline_pd_model, create_pkpd_models
from .tmdd import create_qss_models, create_remaining_models

TYPES = frozenset(('pkpd', 'drug_metabolite', 'tmdd'))
routes = frozenset(('iv', 'oral', 'ivoral'))


def create_workflow(
    type: str,
    route: str = 'oral',
    search_space: Optional[str] = None,
    b_init: Optional[Union[int, float]] = None,
    emax_init: Optional[Union[int, float]] = None,
    ec50_init: Optional[Union[int, float]] = None,
    met_init: Optional[Union[int, float]] = None,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
    extra_model: Optional[Model] = None,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs >= 0)",
):
    """Run the structsearch tool. For more details, see :ref:`structsearch`.

    Parameters
    ----------
    type : str
        Type of model. Currently only 'drug_metabolite' and 'pkpd'
    route : str
        Type of administration. Currently 'oral', 'iv' and 'ivoral'
    search_space : str
        Search space to test
    b_init: float
        Initial estimate for the baseline for pkpd models. The default value is 0.1
    emax_init: float
        Initial estimate for E_MAX (for pkpd models only). The default value is 0.1
    ec50_init: float
        Initial estimate for EC_50 (for pkpd models only). The default value is 0.1
    met_init: float
        Initial estimate for MET (for pkpd models only). The default value is 0.1
    results : ModelfitResults
        Results for the start model
    model : Model
        Pharmpy start model
    extra_model : Model
        Optional extra Pharmpy model to use in TMDD structsearch
    strictness : str or None
        Strictness criteria

    Returns
    -------
    StructSearchResult
        structsearch tool result object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> from pharmpy.tools import run_structsearch, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> run_structsearch(model_type='pkpd', results=results, model=model)   # doctest: +SKIP
    """

    wb = WorkflowBuilder(name="structsearch")
    if type == 'tmdd':
        start_task = Task('run_tmdd', run_tmdd, model, extra_model, strictness)
    elif type == 'pkpd':
        start_task = Task(
            'run_pkpd',
            run_pkpd,
            model,
            search_space,
            b_init,
            emax_init,
            ec50_init,
            met_init,
            strictness,
        )
    elif type == 'drug_metabolite':
        start_task = Task('run_drug_metabolite', run_drug_metabolite, model, route, strictness)
    wb.add_task(start_task)
    return Workflow(wb)


def run_tmdd(context, model, extra_model, strictness):
    model = update_initial_estimates(model)
    if extra_model is not None:
        extra_model = update_initial_estimates(extra_model)
        qss_candidate_models = create_qss_models(
            model, model.modelfit_results.parameter_estimates
        ) + create_qss_models(extra_model, model.modelfit_results.parameter_estimates, index=9)
    else:
        qss_candidate_models = create_qss_models(model, model.modelfit_results.parameter_estimates)

    wf = create_fit_workflow(qss_candidate_models)
    wb = WorkflowBuilder(wf)
    task_results = Task('results', bundle_results)
    wb.add_task(task_results, predecessors=wf.output_tasks)
    qss_run_models = call_workflow(Workflow(wb), 'results_QSS', context)

    ofvs = [
        m.modelfit_results.ofv if m.modelfit_results is not None else np.nan for m in qss_run_models
    ]
    minindex = ofvs.index(np.nanmin(ofvs))
    best_qss_model = qss_run_models[minindex]
    best_qss_model = best_qss_model.replace(parent_model=best_qss_model.name)

    models = create_remaining_models(
        model,
        best_qss_model.modelfit_results.parameter_estimates,
        best_qss_model.name,
        len(best_qss_model.statements.ode_system.find_peripheral_compartments()),
    )
    wf2 = create_fit_workflow(models)
    wb2 = WorkflowBuilder(wf2)
    task_results = Task('results', bundle_results)
    wb2.add_task(task_results, predecessors=wf2.output_tasks)
    run_models = call_workflow(Workflow(wb2), 'results_remaining', context)

    summary_input = summarize_modelfit_results(model.modelfit_results)
    summary_candidates = summarize_modelfit_results(
        [model.modelfit_results for model in qss_run_models + run_models]
    )

    return create_results(
        StructSearchResults,
        model,
        best_qss_model,
        list(run_models),
        rank_type='bic',
        cutoff=None,
        summary_models=pd.concat([summary_input, summary_candidates], keys=[0, 1], names=['step']),
        strictness=strictness,
    )


def run_pkpd(context, model, search_space, b_init, emax_init, ec50_init, met_init, strictness):
    baseline_pd_model = create_baseline_pd_model(
        model, model.modelfit_results.parameter_estimates, b_init
    )
    wf = create_fit_workflow(baseline_pd_model)
    wb = WorkflowBuilder(wf)
    task_results = Task('results2', bundle_results)
    wb.add_task(task_results, predecessors=wf.output_tasks)
    pd_baseline_fit = call_workflow(Workflow(wb), 'results_remaining', context)

    pkpd_models = create_pkpd_models(
        model,
        search_space,
        b_init,
        model.modelfit_results.parameter_estimates,
        emax_init,
        ec50_init,
        met_init,
    )

    wf2 = create_fit_workflow(pkpd_models)
    wb2 = WorkflowBuilder(wf2)
    task_results = Task('results2', bundle_results)
    wb2.add_task(task_results, predecessors=wf2.output_tasks)
    pkpd_models_fit = call_workflow(Workflow(wb2), 'results_remaining', context)

    summary_input = summarize_modelfit_results(model.modelfit_results)
    summary_candidates = summarize_modelfit_results(
        [model.modelfit_results for model in pd_baseline_fit + pkpd_models_fit]
    )

    return create_results(
        StructSearchResults,
        model,
        pd_baseline_fit[0],
        list(pkpd_models_fit),
        rank_type='bic',
        cutoff=None,
        summary_models=pd.concat([summary_input, summary_candidates], keys=[0, 1], names=["step"]),
        strictness=strictness,
    )


def run_drug_metabolite(context, model, route, strictness):
    model = update_initial_estimates(model)
    base_drug_metabolite = create_base_metabolite(model)
    candidate_drug_metabolite = create_drug_metabolite_models(model, route)

    # Run workflow for base model
    wf = create_fit_workflow(base_drug_metabolite)
    wb = WorkflowBuilder(wf)
    task_results = Task('results', bundle_results)
    wb.add_task(task_results, predecessors=wf.output_tasks)
    base_drug_metabolite_fit = call_workflow(wb, 'results_remaining', context)

    # Run workflow for candidate models
    wf = create_fit_workflow(candidate_drug_metabolite)
    wb = WorkflowBuilder(wf)
    task_results = Task('results', bundle_results)
    wb.add_task(task_results, predecessors=wf.output_tasks)
    drug_metabolite_models_fit = call_workflow(wb, 'results_remaining', context)

    summary_input = summarize_modelfit_results(model.modelfit_results)
    summary_candidates = summarize_modelfit_results(
        [model.modelfit_results for model in base_drug_metabolite_fit + drug_metabolite_models_fit]
    )

    return create_results(
        StructSearchResults,
        model,
        base_drug_metabolite_fit[0],
        list(drug_metabolite_models_fit),
        rank_type='bic',
        cutoff=None,
        summary_models=pd.concat(
            [summary_input, summary_candidates],
            keys=[0, 1],
            names=['step'],
        ),
        strictness=strictness,
    )


def bundle_results(*args):
    return args


def _results(model):
    return model.modelfit_results


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    type,
    strictness,
    model,
    route,
):
    if type not in TYPES:
        raise ValueError(f'Invalid `type`: got `{type}`, must be one of {sorted(TYPES)}.')
    if route not in routes:
        raise ValueError(f'Invalid `route`: got `{route}`, must be one of {sorted(routes)}.')

    if strictness is not None and "rse" in strictness.lower():
        if model.estimation_steps[-1].parameter_uncertainty_method is None:
            raise ValueError(
                'parameter_uncertainty_method not set for model, cannot calculate relative standard errors.'
            )


@dataclass(frozen=True)
class StructSearchResults(ToolResults):
    pass
