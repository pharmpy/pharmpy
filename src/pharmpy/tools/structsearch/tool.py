from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pharmpy.deps import numpy as np
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.results import ModelfitResults
from pharmpy.tools import summarize_modelfit_results
from pharmpy.tools.common import ToolResults
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow, call_workflow

from .tmdd import create_qss_models, create_remaining_models

ROUTES = frozenset(('iv', 'oral'))
TYPES = frozenset(('tmdd',))


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
        Type of model. Currently only 'tmdd'
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

    wf = Workflow()
    wf.name = 'structsearch'
    start_task = Task('run_tmdd', run_tmdd, model)
    wf.add_task(start_task)
    results_task = Task('results', _results)
    wf.add_task(results_task, predecessors=[start_task])
    return wf


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

    summary_models = summarize_modelfit_results(
        [model.modelfit_results for model in qss_run_models + run_models]
    )

    res = StructSearchResults(summary_models=summary_models)

    return res


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
