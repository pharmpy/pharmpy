from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Iterable, List, Optional, Union

from pharmpy.basic import Expr
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import add_allometry, get_pk_parameters
from pharmpy.tools import summarize_individuals, summarize_individuals_count_table
from pharmpy.tools.common import ToolResults, update_initial_estimates
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import summarize_errors_from_entries, summarize_modelfit_results_from_entries
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults


def create_workflow(
    model: Optional[Model] = None,
    results: Optional[ModelfitResults] = None,
    allometric_variable: Union[str, Expr] = 'WT',
    reference_value: Union[str, int, float, Expr] = 70,
    parameters: Optional[List[Union[str, Expr]]] = None,
    initials: Optional[List[Union[int, float]]] = None,
    lower_bounds: Optional[List[Union[int, float]]] = None,
    upper_bounds: Optional[List[Union[int, float]]] = None,
    fixed: bool = True,
):
    """Run allometry tool. For more details, see :ref:`allometry`.

    Parameters
    ----------
    model : Model
        Pharmpy model
    results : ModelfitResults
        Results for model
    allometric_variable : str
        Name of the variable to use for allometric scaling (default is WT)
    reference_value : float
        Reference value for the allometric variable (default is 70)
    parameters : list
        Parameters to apply scaling to (default is all CL, Q and V parameters)
    initials : list
        Initial estimates for the exponents. (default is to use 0.75 for CL and Qs and 1 for Vs)
    lower_bounds : list
        Lower bounds for the exponents. (default is 0 for all parameters)
    upper_bounds : list
        Upper bounds for the exponents. (default is 2 for all parameters)
    fixed : bool
        Should the exponents be fixed or not. (default True)

    Returns
    -------
    AllometryResults
        Allometry tool result object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> from pharmpy.tools import run_allometry, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> run_allometry(model=model, results=results, allometric_variable='WGT') # doctest: +SKIP

    """

    wb = WorkflowBuilder(name="allometry")
    if model is not None:
        start_task = Task('start_allometry', start, results, model)
    else:
        start_task = Task('start_allometry', start)
    _add_allometry = partial(
        _add_allometry_on_model,
        allometric_variable=allometric_variable,
        reference_value=reference_value,
        parameters=parameters,
        initials=initials,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        fixed=fixed,
    )
    task_add_allometry = Task('add allometry', _add_allometry)
    wb.add_task(task_add_allometry, predecessors=start_task)
    fit_wf = create_fit_workflow(n=1)
    wb.insert_workflow(fit_wf, predecessors=task_add_allometry)
    results_task = Task('results', globals()['results'])
    wb.add_task(results_task, predecessors=[start_task] + fit_wf.output_tasks)
    return Workflow(wb)


def start(modelfit_results, model_or_model_entry):
    if isinstance(model_or_model_entry, ModelEntry):
        return model_or_model_entry
    log = modelfit_results.log if modelfit_results else None
    model_entry = ModelEntry(model_or_model_entry, modelfit_results=modelfit_results, log=log)
    return model_entry


def _add_allometry_on_model(
    input_model_entry,
    allometric_variable,
    reference_value,
    parameters,
    initials,
    lower_bounds,
    upper_bounds,
    fixed,
):
    input_model, input_res = input_model_entry.model, input_model_entry.modelfit_results
    model = update_initial_estimates(input_model, input_res)
    model = add_allometry(
        model,
        allometric_variable=allometric_variable,
        reference_value=reference_value,
        parameters=parameters,
        initials=initials,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        fixed=fixed,
    )

    model = model.replace(name="scaled_model", description="Allometry model")
    model_entry = ModelEntry(model, parent=input_model)
    return model_entry


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    model,
    allometric_variable,
    parameters,
):
    if model is not None:
        validate_allometric_variable(model, allometric_variable)
        validate_parameters(model, parameters)


def _parse_fs(expr: str):
    return map(str, Expr(expr).free_symbols)


def validate_allometric_variable(model: Model, allometric_variable: str):
    if not set(_parse_fs(allometric_variable)).issubset(model.datainfo.names):
        raise ValueError(
            f'Invalid `allometric_variable`: got `{allometric_variable}`,'
            f' free symbols must be a subset of {sorted(model.datainfo.names)}.'
        )


def validate_parameters(model: Model, parameters: Optional[Iterable[Union[str, Expr]]]):
    if parameters is not None:
        allowed_parameters = set(get_pk_parameters(model)).union(
            str(statement.symbol) for statement in model.statements.before_odes
        )
        if not set().union(*map(_parse_fs, parameters)).issubset(allowed_parameters):
            raise ValueError(
                f'Invalid `parameters`: got `{parameters}`,'
                f' must be NULL/None or'
                f' free symbols must be a subset of {sorted(allowed_parameters)}.'
            )


def results(start_model_entry, allometry_model_entry):
    start_model = start_model_entry.model
    start_res = start_model_entry.modelfit_results
    allometry_model = allometry_model_entry.model
    allometry_res = allometry_model_entry.modelfit_results

    allometry_model_failed = allometry_res is None
    best_model = start_model if allometry_model_failed else allometry_model

    summod = summarize_modelfit_results_from_entries([start_model_entry, allometry_model_entry])
    summod['step'] = [0, 1]
    summods = summod.reset_index().set_index(['step', 'model'])
    suminds = summarize_individuals([start_model, allometry_model], [start_res, allometry_res])
    sumcount = summarize_individuals_count_table(df=suminds)
    sumerrs = summarize_errors_from_entries([start_model_entry, allometry_model_entry])

    return AllometryResults(
        summary_models=summods,
        summary_individuals=suminds,
        summary_individuals_count=sumcount,
        summary_errors=sumerrs,
        final_model=best_model,
        final_results=allometry_res,
    )


@dataclass(frozen=True)
class AllometryResults(ToolResults):
    pass
