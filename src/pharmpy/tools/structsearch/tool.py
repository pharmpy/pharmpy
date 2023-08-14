from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.results import ModelfitResults
from pharmpy.tools import summarize_modelfit_results
from pharmpy.tools.common import ToolResults, create_results
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow, WorkflowBuilder, call_workflow

from .pkpd import create_baseline_pd_model, create_pkpd_models
from .tmdd import create_qss_models, create_remaining_models

ROUTES = frozenset(('iv', 'oral'))
TYPES = frozenset(('tmdd', 'pkpd'))


def create_workflow(
    route: str,
    type: str,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
):
    """Run the structsearch tool. For more details, see :ref:`structsearch`.

    Parameters
    ----------
    route : str
        Route of administration. Either 'pk' or 'oral'
    type : str
        Type of model. Currently only 'tmdd' and 'pkpd'
    results : ModelfitResults
        Results for the start model
    model : Model
        Pharmpy start model

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
    >>> run_structsearch(model_type='tmdd', results=results, model=model)   # doctest: +SKIP
    """

    wb = WorkflowBuilder(name="structsearch")
    if type == 'tmdd':
        start_task = Task('run_tmdd', run_tmdd, model)
    elif type == 'pkpd':
        start_task = Task('run_pkpd', run_pkpd, model)
    wb.add_task(start_task)
    return Workflow(wb)


def run_tmdd(context, model):
    qss_candidate_models = create_qss_models(model)

    wf = create_fit_workflow(qss_candidate_models)
    task_results = Task('results', bundle_results)
    wf.add_task(task_results, predecessors=wf.output_tasks)
    qss_run_models = call_workflow(wf, 'results_QSS', context)

    ofvs = [m.modelfit_results.ofv for m in qss_run_models]
    minindex = ofvs.index(np.nanmin(ofvs))
    best_qss_model = qss_candidate_models[minindex]

    models = create_remaining_models(model, best_qss_model.modelfit_results.parameter_estimates)
    wf2 = create_fit_workflow(models)
    task_results = Task('results', bundle_results)
    wf.add_task(task_results, predecessors=wf2.output_tasks)
    run_models = call_workflow(wf, 'results_remaining', context)

    summary_input = summarize_modelfit_results(model.modelfit_results)
    summary_candidates = summarize_modelfit_results(
        [model.modelfit_results for model in qss_run_models + run_models]
    )

    return create_results(
        StructSearchResults,
        model,
        model,
        qss_run_models + run_models,
        rank_type='bic',
        cutoff=None,
        summary_models=pd.concat([summary_input, summary_candidates], keys=[0, 1], names=['step']),
    )


def run_pkpd(context, model):
    baseline_pd_model = create_baseline_pd_model(model, model.modelfit_results.parameter_estimates)
    wf = create_fit_workflow(baseline_pd_model)
    wb = WorkflowBuilder(wf)
    task_results = Task('results2', bundle_results)
    wb.add_task(task_results, predecessors=wf.output_tasks)
    pd_baseline_fit = call_workflow(Workflow(wb), 'results_remaining', context)

    pkpd_models = create_pkpd_models(
        model,
        pd_baseline_fit[0].modelfit_results.parameter_estimates,
        model.modelfit_results.parameter_estimates,
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
        model,
        list(pd_baseline_fit + pkpd_models_fit),
        rank_type='bic',
        cutoff=None,
        summary_models=pd.concat([summary_input, summary_candidates], keys=[0, 1], names=["step"]),
    )


def bundle_results(*args):
    return args


def _results(model):
    return model.modelfit_results


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    route,
    type,
):
    if route not in ROUTES:
        raise ValueError(f'Invalid `route`: got `{route}`, must be one of {sorted(ROUTES)}.')

    if type not in TYPES:
        raise ValueError(f'Invalid `type`: got `{type}`, must be one of {sorted(TYPES)}.')


@dataclass(frozen=True)
class StructSearchResults(ToolResults):
    pass
