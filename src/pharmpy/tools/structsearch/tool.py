from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.results import ModelfitResults
from pharmpy.tools.common import ToolResults
from pharmpy.workflows import Task, Workflow

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


def run_qss(model):
    return model


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
