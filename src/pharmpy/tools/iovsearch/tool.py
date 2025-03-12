from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Literal, Optional, TypeVar, Union

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
from pharmpy.tools.run import calculate_mbic_penalty, summarize_modelfit_results_from_entries
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults

NAME_WF = 'iovsearch'

T = TypeVar('T')


def create_workflow(
    model: Model,
    results: ModelfitResults,
    column: str = 'OCC',
    list_of_parameters: Optional[list[Union[str, list[str]]]] = None,
    rank_type: Literal[tuple(RANK_TYPES)] = 'bic',
    cutoff: Optional[Union[float, int]] = None,
    distribution: Literal[tuple(ADD_IOV_DISTRIBUTION)] = 'same-as-iiv',
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    E: Optional[Union[float, str]] = None,
):
    """Run IOVsearch tool. For more details, see :ref:`iovsearch`.

    Parameters
    ----------
    model : Model
        Pharmpy model
    results : ModelfitResults
        Results for model
    column : str
        Name of column in dataset to use as occasion column (default is 'OCC')
    list_of_parameters : None or list
        List of parameters to test IOV on, if none all parameters with IIV will be tested (default)
    rank_type : {'ofv', 'lrt', 'aic', 'bic', 'mbic'}
        Which ranking type should be used. Default is BIC.
    cutoff : None or float
        Cutoff for which value of the ranking type that is considered significant. Default
        is None (all models will be ranked)
    distribution : {'disjoint', 'joint', 'explicit', 'same-as-iiv'}
        Which distribution added IOVs should have (default is same-as-iiv)
    strictness : str
        Strictness criteria
    E : float
        Expected number of predictors (used for mBIC). Must be set when using mBIC

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
    >>> run_iovsearch(model=model, results=results, column='OCC')      # doctest: +SKIP
    """

    wb = WorkflowBuilder(name=NAME_WF)

    init_task = create_init(model, results)
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
        E,
        strictness,
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
        E,
        strictness,
    )

    wb.add_task(results_task, predecessors=search_output)

    return Workflow(wb)


def create_init(model, modelfit_results):
    if model is None:
        task = Task('init', _init, model)
    else:
        task = Task('init', _init, modelfit_results, model)
    return task


def _init(context, modelfit_results, model):
    context.log_info("Starting tool iovsearch")
    model = model.replace(name="input", description="")
    me = ModelEntry.create(model, modelfit_results=modelfit_results)
    return me


def task_brute_force_search(
    context,
    occ: str,
    list_of_parameters: Union[None, list],
    rank_type: str,
    cutoff: Union[None, float],
    bic_type: Union[None, str],
    E: Optional[float],
    strictness: str,
    distribution: str,
    input_model_entry: ModelEntry,
):
    # Create links to input model
    context.store_input_model_entry(input_model_entry)

    input_model = input_model_entry.model
    # NOTE: Default is to try all IIV ETAs.
    list_of_parameters = prepare_list_of_parameters(input_model, list_of_parameters)

    current_step = 0
    step_mapping = {current_step: [input_model.name]}

    # NOTE: Check that model has at least one IIV.
    if not list_of_parameters:
        return step_mapping, [input_model_entry]

    model_with_iov_entry = create_iov_base_model_entry(
        input_model_entry, occ, list_of_parameters, distribution
    )
    wf = create_fit_workflow(modelentries=[model_with_iov_entry])
    model_with_iov_entry = context.call_workflow(wf, f'{NAME_WF}-fit-with-matching-IOVs')
    model_with_iov = model_with_iov_entry.model

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
    iov_candidate_entries = context.call_workflow(wf, f'{NAME_WF}-fit-with-removed-IOVs')

    if rank_type == "mbic":
        ref = model_with_iov
        penalties = [
            calculate_mbic_penalty(me.model, ['iov'], base_model=ref, E_p=E)
            for me in [input_model_entry, model_with_iov_entry, *iov_candidate_entries]
        ]
    else:
        penalties = None

    # NOTE: Keep the best candidate.
    best_model_entry_so_far = get_best_model(
        input_model_entry,
        [model_with_iov_entry, *iov_candidate_entries],
        rank_type=rank_type,
        penalties=penalties,
        cutoff=cutoff,
        bic_type=bic_type,
        strictness=strictness,
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
    iiv_candidate_entries = context.call_workflow(wf, f'{NAME_WF}-fit-with-removed-IIVs')
    current_step += 1
    step_mapping[current_step] = [model_entry.model.name for model_entry in iiv_candidate_entries]

    return step_mapping, [
        input_model_entry,
        model_with_iov_entry,
        *iov_candidate_entries,
        *iiv_candidate_entries,
    ]


def prepare_list_of_parameters(input_model, list_of_parameters):
    # NOTE: Default is to try all IIV ETAs.
    if list_of_parameters is None:
        iiv = _get_nonfixed_iivs(input_model)
        iiv_before_odes = iiv.free_symbols.intersection(
            input_model.statements.before_odes.free_symbols
        )
        list_of_parameters = [iiv.name for iiv in iiv_before_odes]
    return list_of_parameters


def create_iov_base_model_entry(input_model_entry, occ, list_of_parameters, distribution):
    # NOTE: Add IOVs on given parameters or all parameters with IIVs.
    input_model, input_res = input_model_entry.model, input_model_entry.modelfit_results
    model_with_iov = input_model.replace(name='iovsearch_run1')
    model_with_iov = update_initial_estimates(model_with_iov, input_res)
    # TODO: Should we exclude already present IOVs?
    model_with_iov = add_iov(model_with_iov, occ, list_of_parameters, distribution=distribution)
    model_with_iov = model_with_iov.replace(description=_create_description(model_with_iov))
    return ModelEntry.create(model_with_iov, parent=input_model)


def _create_description(model):
    iiv_desc = pharmpy.tools.iivsearch.algorithms.create_description(model)
    iov_desc = pharmpy.tools.iivsearch.algorithms.create_description(model, iov=True)
    return f'IIV({iiv_desc});IOV({iov_desc})'


def create_candidate_model_entry(
    remove_func: Callable[[Model, list[str]], None],
    model_entry: ModelEntry,
    subset: list[str],
    n: int,
):
    parent_model, parent_res = model_entry.model, model_entry.modelfit_results
    model_with_some_etas_removed = parent_model.replace(name=f'iovsearch_run{n}')
    model_with_some_etas_removed = update_initial_estimates(
        model_with_some_etas_removed, parent_res
    )
    model_with_some_etas_removed = remove_func(model_with_some_etas_removed, subset)
    model_with_some_etas_removed = model_with_some_etas_removed.replace(
        description=_create_description(model_with_some_etas_removed)
    )
    return ModelEntry.create(model_with_some_etas_removed, parent=parent_model)


def wf_etas_removal(
    remove: Callable[[Model, list[str]], None],
    model_entry: ModelEntry,
    etas_subsets: Iterable[tuple[str]],
    i: int,
):
    wb = WorkflowBuilder()
    j = i
    for subset_of_iiv_parameters in etas_subsets:
        task = Task(
            repr(subset_of_iiv_parameters),
            create_candidate_model_entry,
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


def get_best_model(
    base_entry: ModelEntry,
    model_entries: list[ModelEntry],
    rank_type: str,
    penalties: Union[None, list[float]],
    cutoff: Union[None, float],
    bic_type: Union[None, str],
    strictness: str,
):
    candidate_entries = [base_entry, *model_entries]
    df = summarize_tool(
        model_entries,
        base_entry,
        rank_type=rank_type,
        penalties=penalties,
        cutoff=cutoff,
        bic_type=bic_type,
        strictness=strictness,
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


def task_results(
    context, rank_type, cutoff, bic_type, E, strictness, step_mapping_and_model_entries
):
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
                candidate_entries,
                ref_model_entry,
                rank_type=rank_type,
                cutoff=cutoff,
                bic_type=bic_type,
                strictness=strictness,
            )
            sum_tool.append(sum_tool_step)

    keys = list(range(1, len(step_mapping)))

    if rank_type == "mbic":
        models = [me.model for me in [base_model_entry] + res_model_entries]
        ref = sorted(models, key=lambda model: len(model.parameters), reverse=True)[0]
        penalties = [
            calculate_mbic_penalty(me.model, ['iov'], base_model=ref, E_p=E)
            for me in [base_model_entry] + res_model_entries
        ]
    else:
        penalties = None

    res = create_results(
        IOVSearchResults,
        base_model_entry,
        base_model_entry,
        res_model_entries,
        rank_type,
        cutoff,
        bic_type=bic_type,
        penalties=penalties,
        summary_models=pd.concat(sum_mod, keys=[0] + keys, names=['step']),
        strictness=strictness,
        context=context,
    )

    # NOTE: This overwrites the default summary_tool field
    res = replace(res, summary_tool=pd.concat(sum_tool, keys=keys, names=['step']))

    final_model = res.final_model.replace(name="final")
    context.store_final_model_entry(final_model)

    context.log_info("Finisihing tool iovsearch")
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
    E,
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
    if rank_type != 'mbic' and E is not None:
        raise ValueError(f'E can only be provided when `rank_type` is mbic: got `{rank_type}`')
    if rank_type == 'mbic':
        if E is None:
            raise ValueError('Value `E` must be provided when using mbic')
        if isinstance(E, float) and E <= 0.0:
            raise ValueError(f'Value `E` must be more than 0: got `{E}`')
        if isinstance(E, str) and not E.endswith('%'):
            raise ValueError(f'Value `E` must be denoted with `%`: got `{E}`')


@dataclass(frozen=True)
class IOVSearchResults(ToolResults):
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

    for statement in model.statements.before_odes:
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
