from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

from pharmpy.deps import numpy as np
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import set_initial_estimates, set_name, set_tmdd
from pharmpy.results import ModelfitResults
from pharmpy.tools.common import ToolResults
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow, call_workflow

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
    start_task = Task('run_qss', run_qss, model)
    wf.add_task(start_task)
    results_task = Task('results', _results)
    wf.add_task(results_task, predecessors=[start_task])
    return wf


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def run_qss(context, model):
    qss_base_model = set_tmdd(model, type="QSS")
    all_inits = product_dict(POP_KDEG=(0.5623, 17.28), POP_R_0=(0.001, 0.01, 0.1, 1))
    qss_candidate_models = [
        set_initial_estimates(set_name(qss_base_model, f"QSS{i}"), inits)
        for i, inits in enumerate(all_inits, start=1)
    ]
    wf = create_fit_workflow(qss_candidate_models)
    task_results = Task('results', bundle_results)
    wf.add_task(task_results, predecessors=wf.output_tasks)
    results = call_workflow(wf, 'results_QSS', context)
    ofvs = [res.ofv for res in results]
    minindex = ofvs.index(np.nanmin(ofvs))
    return qss_candidate_models[minindex]


def bundle_results(*args):
    return [model.modelfit_results for model in args]


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
