from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Optional, Tuple, TypeVar, Union

import pharmpy.tools.iivsearch.algorithms
from pharmpy.basic import Expr
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.internals.set.subsets import non_empty_proper_subsets, non_empty_subsets
from pharmpy.model import Assignment, Model, RandomVariables
from pharmpy.modeling import add_iov, get_omegas, get_pk_parameters, remove_iiv, remove_iov
from pharmpy.modeling.parameter_variability import ADD_IOV_DISTRIBUTION
from pharmpy.tools.common import (
    RANK_TYPES,
    ToolResults,
    create_results,
    summarize_tool,
    update_initial_estimates,
)
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import summarize_modelfit_results_from_entries
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder, call_workflow
from pharmpy.workflows.results import ModelfitResults

NAME_WF = 'iovsearch'

T = TypeVar('T')


def create_workflow(
    column: str = 'OCC',
    list_of_parameters: Optional[List[Union[str, List[str]]]] = None,
    rank_type: Literal[tuple(RANK_TYPES)] = 'mbic',
    cutoff: Optional[Union[float, int]] = None,
    distribution: Literal[tuple(ADD_IOV_DISTRIBUTION)] = 'same-as-iiv',
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
):
    """Run IOVsearch tool. For more details, see :ref:`iovsearch`.

    Parameters
    ----------
    column : str
        Name of column in dataset to use as occasion column (default is 'OCC')
    list_of_parameters : None or list
        List of parameters to test IOV on, if none all parameters with IIV will be tested (default)
    rank_type : {'ofv', 'lrt', 'aic', 'bic', 'mbic'}
        Which ranking type should be used. Default is mBIC.
    cutoff : None or float
        Cutoff for which value of the ranking type that is considered significant. Default
        is None (all models will be ranked)
    distribution : {'disjoint', 'joint', 'explicit', 'same-as-iiv'}
        Which distribution added IOVs should have (default is same-as-iiv)
    results : ModelfitResults
        Results for model
    model : Model
        Pharmpy model
    strictness : str or None
        Strictness criteria

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

    wb = WorkflowBuilder(name=NAME_WF)

    init_task = init(model, results)
    wb.add_task(init_task)

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

    wb.add_task(search_task, predecessors=init_task)
    search_output = wb.output_tasks

    results_task = Task(
        'results',
        task_results,
        rank_type,
        cutoff,
        bic_type,
        strictness,
    )

    wb.add_task(results_task, predecessors=search_output)

    return Workflow(wb)


def init(model, modelfit_results):
    return (
        Task('init', _model_entry, model)
        if model is None
        else Task('init', _model_entry, modelfit_results, model)
    )


def _model_entry(modelfit_results, model):
    return ModelEntry.create(model, modelfit_results=modelfit_results)


def task_brute_force_search(
    context,
    occ: str,
    list_of_parameters: Union[None, list],
    rank_type: str,
    cutoff: Union[None, float],
    bic_type: Union[None, str],
    distribution: str,
    input_model_entry: ModelEntry,
):
    input_model, input_res = input_model_entry.model, input_model_entry.modelfit_results
    # NOTE: Default is to try all IIV ETAs.
    if list_of_parameters is None:
        iiv = _get_nonfixed_iivs(input_model)
        iiv_before_odes = iiv.free_symbols.intersection(
            input_model.statements.before_odes.free_symbols
        )
        list_of_parameters = [iiv.name for iiv in iiv_before_odes]

    current_step = 0
    step_mapping = {current_step: [input_model.name]}

    # NOTE: Check that model has at least one IIV.
    if not list_of_parameters:
        return step_mapping, [input_model_entry]

    # NOTE: Add IOVs on given parameters or all parameters with IIVs.
    name = 'iovsearch_run1'
    model_with_iov = input_model.replace(name=name)
    model_with_iov = update_initial_estimates(model_with_iov, input_res)
    # TODO: Should we exclude already present IOVs?
    model_with_iov = add_iov(model_with_iov, occ, list_of_parameters, distribution=distribution)
    model_with_iov = model_with_iov.replace(description=_create_description(model_with_iov))
    # NOTE: Fit the new model.
    model_with_iov_entry = ModelEntry.create(model_with_iov, parent=input_model)
    wf = create_fit_workflow(modelentries=[model_with_iov_entry])
    model_with_iov_entry = call_workflow(wf, f'{NAME_WF}-fit-with-matching-IOVs', context)

    # NOTE: Remove IOVs. Test all subsets (~2^n).
    # TODO: Should we exclude already present IOVs?
    iov = model_with_iov.random_variables.iov
    # NOTE: We only need to remove the IOV ETA corresponding to the first
    # category in order to remove all IOV ETAs of the other categories
    all_iov_parameters = list(filter(lambda name: name.endswith('_1'), iov.names))
    no_of_models = 1
    wf = wf_etas_removal(
        remove_iov,
        model_with_iov_entry,
        non_empty_proper_subsets(all_iov_parameters),
        no_of_models + 1,
    )
    iov_candidate_entries = call_workflow(wf, f'{NAME_WF}-fit-with-removed-IOVs', context)

    # NOTE: Keep the best candidate.
    best_model_entry_so_far = best_model(
        input_model_entry,
        [model_with_iov_entry, *iov_candidate_entries],
        rank_type=rank_type,
        cutoff=cutoff,
        bic_type=bic_type,
    )

    current_step += 1
    step_mapping[current_step] = [model_with_iov.name] + [
        model_entry.model.name for model_entry in iov_candidate_entries
    ]

    # NOTE: If no improvement with respect to input model, STOP.
    if best_model_entry_so_far.model is input_model:
        return step_mapping, [input_model_entry, model_with_iov_entry, *iov_candidate_entries]

    # NOTE: Remove IIV with corresponding IOVs. Test all subsets (~2^n).
    iiv_parameters_with_associated_iov = list(
        map(
            lambda s: s.name,
            _get_iiv_etas_with_corresponding_iov(best_model_entry_so_far.model),
        )
    )
    # TODO: Should we exclude already present IOVs?
    no_of_models = len(iov_candidate_entries) + 1
    wf = wf_etas_removal(
        remove_iiv,
        best_model_entry_so_far,
        non_empty_subsets(iiv_parameters_with_associated_iov),
        no_of_models + 1,
    )
    iiv_candidate_entries = call_workflow(wf, f'{NAME_WF}-fit-with-removed-IIVs', context)
    current_step += 1
    step_mapping[current_step] = [model_entry.model.name for model_entry in iiv_candidate_entries]

    return step_mapping, [
        input_model_entry,
        model_with_iov_entry,
        *iov_candidate_entries,
        *iiv_candidate_entries,
    ]


def _create_description(model):
    iiv_desc = pharmpy.tools.iivsearch.algorithms.create_description(model)
    iov_desc = pharmpy.tools.iivsearch.algorithms.create_description(model, iov=True)
    return f'IIV({iiv_desc});IOV({iov_desc})'


def task_remove_etas_subset(
    remove: Callable[[Model, List[str]], None], model_entry: ModelEntry, subset: List[str], n: int
):
    parent_model, parent_res = model_entry.model, model_entry.modelfit_results
    model_with_some_etas_removed = parent_model.replace(name=f'iovsearch_run{n}')
    model_with_some_etas_removed = update_initial_estimates(
        model_with_some_etas_removed, parent_res
    )
    model_with_some_etas_removed = remove(model_with_some_etas_removed, subset)
    model_with_some_etas_removed = model_with_some_etas_removed.replace(
        description=_create_description(model_with_some_etas_removed)
    )
    return ModelEntry.create(model_with_some_etas_removed, parent=parent_model)


def wf_etas_removal(
    remove: Callable[[Model, List[str]], None],
    model_entry: ModelEntry,
    etas_subsets: Iterable[Tuple[str]],
    i: int,
):
    wb = WorkflowBuilder()
    j = i
    for subset_of_iiv_parameters in etas_subsets:
        task = Task(
            repr(subset_of_iiv_parameters),
            task_remove_etas_subset,
            remove,
            model_entry,
            list(subset_of_iiv_parameters),
            j,
        )
        wb.add_task(task)
        j += 1

    n = j - i
    wf_fit = create_fit_workflow(n=n)
    wb.insert_workflow(wf_fit)

    task_gather = Task('gather', lambda *model_entries: model_entries)
    wb.add_task(task_gather, predecessors=wb.output_tasks)
    return Workflow(wb)


def best_model(
    base_entry: ModelEntry,
    model_entries: List[ModelEntry],
    rank_type: str,
    cutoff: Union[None, float],
    bic_type: Union[None, str],
):
    candidate_entries = [base_entry, *model_entries]
    df = summarize_tool(
        model_entries, base_entry, rank_type=rank_type, cutoff=cutoff, bic_type=bic_type
    )
    best_model_name = df['rank'].idxmin()

    try:
        return [
            model_entry
            for model_entry in candidate_entries
            if model_entry.model.name == best_model_name
        ][0]
    except IndexError:
        return base_entry


def task_results(context, rank_type, cutoff, bic_type, strictness, step_mapping_and_model_entries):
    step_mapping, (base_model_entry, *res_model_entries) = step_mapping_and_model_entries

    model_dict = {
        model_entry.model.name: model_entry
        for model_entry in [base_model_entry] + res_model_entries
    }
    sum_mod, sum_tool = [], []
    for step, model_names in step_mapping.items():
        candidate_entries = [
            model_entry
            for model_name, model_entry in model_dict.items()
            if model_name in model_names
        ]
        sum_mod_step = summarize_modelfit_results_from_entries(candidate_entries)
        sum_mod.append(sum_mod_step)
        if step >= 1:
            ref_model_entry = model_dict[candidate_entries[0].parent.name]
            sum_tool_step = summarize_tool(
                candidate_entries, ref_model_entry, rank_type, cutoff, bic_type
            )
            sum_tool.append(sum_tool_step)

    keys = list(range(1, len(step_mapping)))

    res = create_results(
        IOVSearchResults,
        base_model_entry,
        base_model_entry,
        res_model_entries,
        rank_type,
        cutoff,
        bic_type=bic_type,
        summary_models=pd.concat(sum_mod, keys=[0] + keys, names=['step']),
        strictness=strictness,
        context=context,
    )

    # NOTE: This overwrites the default summary_tool field
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
    strictness,
):
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
            if not set(_flatten_list(list_of_parameters)).issubset(allowed_parameters):
                raise ValueError(
                    f'Invalid `list_of_parameters`: got `{list_of_parameters}`,'
                    f' must be NULL/None or a subset of {sorted(allowed_parameters)}.'
                )
    if strictness is not None and "rse" in strictness.lower():
        if model.execution_steps[-1].parameter_uncertainty_method is None:
            raise ValueError(
                'parameter_uncertainty_method not set for model, cannot calculate relative standard errors.'
            )


@dataclass(frozen=True)
class IOVSearchResults(ToolResults):
    rst_path = Path(__file__).resolve().parent / 'report.rst'
    pass


def _get_iov_piecewise_assignment_symbols(model: Model):
    iovs = set(Expr.symbol(rv) for rv in model.random_variables.iov.names)
    for statement in model.statements:
        if isinstance(statement, Assignment) and statement.expression.is_piecewise():
            try:
                expression_symbols = [p[0] for p in statement.expression.args]
            except (ValueError, NotImplementedError):
                pass  # NOTE: These exceptions are raised by complex Piecewise
                # statements that can be present in user code.
            else:
                if all(s in iovs for s in expression_symbols):
                    yield statement.symbol


def _get_iiv_etas_with_corresponding_iov(model: Model):
    iovs = set(_get_iov_piecewise_assignment_symbols(model))
    iiv = _get_nonfixed_iivs(model)

    for statement in model.statements:
        if isinstance(statement, Assignment) and statement.expression.is_add():
            for symbol in statement.expression.free_symbols:
                if symbol in iovs:
                    rest = statement.expression - symbol
                    if rest.is_symbol() and rest in iiv:
                        yield rest
                    break


def _get_nonfixed_iivs(model):
    fixed_omegas = get_omegas(model).fixed.names
    iivs = model.random_variables.iiv
    nonfixed_iivs = [
        iiv for iiv in iivs if str(list(iiv.variance.free_symbols)[0]) not in fixed_omegas
    ]
    return RandomVariables.create(nonfixed_iivs)


def _flatten_list(some_list):
    if isinstance(some_list[0], list):
        return [x1 for x2 in some_list for x1 in x2]
    else:
        return some_list
