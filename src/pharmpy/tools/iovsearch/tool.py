from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import pharmpy.tools.iivsearch.algorithms
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.internals.set.subsets import non_empty_proper_subsets, non_empty_subsets
from pharmpy.model import Assignment, Model
from pharmpy.modeling import add_iov, get_pk_parameters, remove_iiv, remove_iov
from pharmpy.modeling.parameter_variability import ADD_IOV_DISTRIBUTION
from pharmpy.modeling.results import RANK_TYPES
from pharmpy.results import ModelfitResults
from pharmpy.tools import rank_models, summarize_modelfit_results
from pharmpy.tools.common import (
    ToolResults,
    create_results,
    summarize_tool,
    update_initial_estimates,
)
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow, call_workflow

NAME_WF = 'iovsearch'

T = TypeVar('T')


def create_workflow(
    column: str = 'OCC',
    list_of_parameters: Optional[List[str]] = None,
    rank_type: str = 'bic',
    cutoff: Optional[Union[float, int]] = None,
    distribution: str = 'same-as-iiv',
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
):
    """Run IOVsearch tool. For more details, see :ref:`iovsearch`.

    Parameters
    ----------
    column : str
        Name of column in dataset to use as occasion column (default is 'OCC')
    list_of_parameters : None or list
        List of parameters to test IOV on, if none all parameters with IIV will be tested (default)
    rank_type : str
        Which ranking type should be used (OFV, AIC, BIC). Default is BIC
    cutoff : None or float
        Cutoff for which value of the ranking type that is considered significant. Default
        is None (all models will be ranked)
    distribution : str
        Which distribution added IOVs should have (default is same-as-iiv)
    results : ModelfitResults
        Results for model
    model : Model
        Pharmpy model

    Returns
    -------
    IOVSearchResults
        IOVSearch tool result object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import run_iovsearch, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> run_iovsearch('OCC', results=results, model=model)      # doctest: +SKIP
    """

    wf = Workflow()
    wf.name = NAME_WF

    init_task = init(model)
    wf.add_task(init_task)

    bic_type = 'random'
    search_task = Task(
        'search',
        task_brute_force_search,
        column,
        list_of_parameters,
        rank_type,
        cutoff,
        bic_type,
        distribution,
    )

    wf.add_task(search_task, predecessors=init_task)
    search_output = wf.output_tasks

    results_task = Task(
        'results',
        task_results,
        rank_type,
        cutoff,
        bic_type,
    )

    wf.add_task(results_task, predecessors=search_output)

    return wf


def init(model):
    return (
        Task('init', lambda model: model)
        if model is None
        else Task('init', lambda model: model, model)
    )


def task_brute_force_search(
    context,
    occ: str,
    list_of_parameters: Union[None, list],
    rank_type: str,
    cutoff: Union[None, float],
    bic_type: Union[None, str],
    distribution: str,
    model: Model,
):
    # NOTE Default is to try all IIV ETAs.
    if list_of_parameters is None:
        iiv = model.random_variables.iiv
        iiv_before_odes = iiv.free_symbols.intersection(model.statements.before_odes.free_symbols)
        list_of_parameters = [iiv.name for iiv in iiv_before_odes]

    current_step = 0
    step_mapping = {current_step: [model.name]}

    # NOTE Check that model has at least one IIV.
    if not list_of_parameters:
        return step_mapping, [model]

    # NOTE Add IOVs on given parameters or all parameters with IIVs.
    name = 'iovsearch_run1'
    model_with_iov = model.replace(name=name, parent_model=model.name)
    model_with_iov = update_initial_estimates(model_with_iov)
    # TODO should we exclude already present IOVs?
    model_with_iov = add_iov(model_with_iov, occ, list_of_parameters, distribution=distribution)
    model_with_iov = model_with_iov.replace(description=_create_description(model_with_iov))
    # NOTE Fit the new model.
    wf = create_fit_workflow(models=[model_with_iov])
    model_with_iov = call_workflow(wf, f'{NAME_WF}-fit-with-matching-IOVs', context)

    # NOTE Remove IOVs. Test all subsets (~2^n).
    # TODO should we exclude already present IOVs?
    iov = model_with_iov.random_variables.iov
    # NOTE We only need to remove the IOV ETA corresponding to the first
    # category in order to remove all IOV ETAs of the other categories
    all_iov_parameters = list(filter(lambda name: name.endswith('_1'), iov.names))
    no_of_models = 1
    wf = wf_etas_removal(
        remove_iov, model_with_iov, non_empty_proper_subsets(all_iov_parameters), no_of_models + 1
    )
    iov_candidates = call_workflow(wf, f'{NAME_WF}-fit-with-removed-IOVs', context)

    # NOTE Keep best candidate.
    best_model_so_far = best_model(
        model,
        [model_with_iov, *iov_candidates],
        rank_type=rank_type,
        cutoff=cutoff,
        bic_type=bic_type,
    )

    current_step += 1
    step_mapping[current_step] = [model_with_iov.name] + [model.name for model in iov_candidates]

    # NOTE If no improvement with respect to input model, STOP.
    if best_model_so_far is model:
        return step_mapping, [model, model_with_iov, *iov_candidates]

    # NOTE Remove IIV with corresponding IOVs. Test all subsets (~2^n).
    iiv_parameters_with_associated_iov = list(
        map(
            lambda s: s.name,
            _get_iiv_etas_with_corresponding_iov(best_model_so_far),
        )
    )
    # TODO should we exclude already present IOVs?
    no_of_models = len(iov_candidates) + 1
    wf = wf_etas_removal(
        remove_iiv,
        best_model_so_far,
        non_empty_subsets(iiv_parameters_with_associated_iov),
        no_of_models + 1,
    )
    iiv_candidates = call_workflow(wf, f'{NAME_WF}-fit-with-removed-IIVs', context)
    current_step += 1
    step_mapping[current_step] = [model.name for model in iiv_candidates]

    return step_mapping, [model, model_with_iov, *iov_candidates, *iiv_candidates]


def _create_description(model):
    iiv_desc = pharmpy.tools.iivsearch.algorithms.create_description(model)
    iov_desc = pharmpy.tools.iivsearch.algorithms.create_description(model, iov=True)
    return f'IIV({iiv_desc});IOV({iov_desc})'


def task_remove_etas_subset(
    remove: Callable[[Model, List[str]], None], model: Model, subset: List[str], n: int
):
    model_with_some_etas_removed = model.replace(name=f'iovsearch_run{n}', parent_model=model.name)
    model_with_some_etas_removed = update_initial_estimates(model_with_some_etas_removed)
    model_with_some_etas_removed = remove(model_with_some_etas_removed, subset)
    model_with_some_etas_removed = model_with_some_etas_removed.replace(
        description=_create_description(model_with_some_etas_removed)
    )
    return model_with_some_etas_removed


def wf_etas_removal(
    remove: Callable[[Model, List[str]], None],
    model: Model,
    etas_subsets: Iterable[Tuple[str]],
    i: int,
):
    wf = Workflow()
    j = i
    for subset_of_iiv_parameters in etas_subsets:
        task = Task(
            repr(subset_of_iiv_parameters),
            task_remove_etas_subset,
            remove,
            model,
            list(subset_of_iiv_parameters),
            j,
        )
        wf.add_task(task)
        j += 1

    n = j - i
    wf_fit = create_fit_workflow(n=n)
    wf.insert_workflow(wf_fit)

    task_gather = Task('gather', lambda *models: models)
    wf.add_task(task_gather, predecessors=wf.output_tasks)
    return wf


def best_model(
    base: Model,
    models: List[Model],
    rank_type: str,
    cutoff: Union[None, float],
    bic_type: Union[None, str],
):
    candidates = [base, *models]
    df = rank_models(base, candidates, rank_type=rank_type, cutoff=cutoff, bic_type=bic_type)
    best_model_name = df['rank'].idxmin()

    try:
        return [model for model in candidates if model.name == best_model_name][0]
    except IndexError:
        return base


def task_results(rank_type, cutoff, bic_type, models):
    step_mapping, (base_model, *res_models) = models

    model_dict = {model.name: model for model in [base_model] + res_models}
    sum_mod, sum_tool = [], []
    for step, model_names in step_mapping.items():
        candidates = [model for model in [base_model] + res_models if model.name in model_names]
        sum_mod_step = summarize_modelfit_results([model.modelfit_results for model in candidates])
        sum_mod.append(sum_mod_step)
        if step >= 1:
            ref_model = model_dict[candidates[0].parent_model]
            sum_tool_step = summarize_tool(candidates, ref_model, rank_type, cutoff, bic_type)
            sum_tool.append(sum_tool_step)

    keys = list(range(1, len(step_mapping)))

    res = create_results(
        IOVSearchResults,
        base_model,
        base_model,
        res_models,
        rank_type,
        cutoff,
        bic_type=bic_type,
        summary_models=pd.concat(sum_mod, keys=[0] + keys, names=['step']),
    )

    # NOTE This overwrites the default summary_tool field
    res = replace(res, summary_tool=pd.concat(sum_tool, keys=keys, names=['step']))

    return res


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    model,
    column,
    list_of_parameters,
    rank_type,
    distribution,
):
    if rank_type not in RANK_TYPES:
        raise ValueError(
            f'Invalid `rank_type`: got `{rank_type}`, must be one of {sorted(RANK_TYPES)}.'
        )

    if distribution not in ADD_IOV_DISTRIBUTION:
        raise ValueError(
            f'Invalid `distribution`: got `{distribution}`,'
            f' must be one of {sorted(ADD_IOV_DISTRIBUTION)}.'
        )

    if model is not None:
        if column not in model.datainfo.names:
            raise ValueError(
                f'Invalid `column`: got `{column}`,'
                f' must be one of {sorted(model.datainfo.names)}.'
            )

        if list_of_parameters is not None:
            allowed_parameters = set(get_pk_parameters(model)).union(
                str(statement.symbol) for statement in model.statements.before_odes
            )
            if not set(list_of_parameters).issubset(allowed_parameters):
                raise ValueError(
                    f'Invalid `list_of_parameters`: got `{list_of_parameters}`,'
                    f' must be NULL/None or a subset of {sorted(allowed_parameters)}.'
                )


@dataclass(frozen=True)
class IOVSearchResults(ToolResults):
    pass


def _get_iov_piecewise_assignment_symbols(model: Model):
    iovs = set(sympy.Symbol(rv) for rv in model.random_variables.iov.names)
    for statement in model.statements:
        if isinstance(statement, Assignment) and isinstance(statement.expression, sympy.Piecewise):
            try:
                expression_symbols = [p[0] for p in statement.expression.as_expr_set_pairs()]
            except (ValueError, NotImplementedError):
                pass  # NOTE These exceptions are raised by complex Piecewise
                # statements that can be present in user code.
            else:
                if all(s in iovs for s in expression_symbols):
                    yield statement.symbol


def _get_iiv_etas_with_corresponding_iov(model: Model):
    iovs = set(_get_iov_piecewise_assignment_symbols(model))
    iiv = model.random_variables.iiv

    for statement in model.statements:
        if isinstance(statement, Assignment) and isinstance(statement.expression, sympy.Add):
            for symbol in statement.expression.free_symbols:
                if symbol in iovs:
                    rest = statement.expression - symbol
                    if isinstance(rest, sympy.Symbol) and rest in iiv:
                        yield rest
                    break
