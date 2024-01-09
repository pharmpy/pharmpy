from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import List, Literal, Optional, Union

import pharmpy.tools.iivsearch.algorithms as algorithms
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import (
    add_pd_iiv,
    add_pk_iiv,
    calculate_bic,
    create_joint_distribution,
    has_random_effect,
)
from pharmpy.tools import summarize_modelfit_results
from pharmpy.tools.common import RANK_TYPES, ToolResults, create_results, update_initial_estimates
from pharmpy.tools.iivsearch.algorithms import _get_fixed_etas
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder, call_workflow
from pharmpy.workflows.results import ModelfitResults

IIV_STRATEGIES = frozenset(
    ('no_add', 'add_diagonal', 'fullblock', 'pd_add_diagonal', 'pd_fullblock')
)
IIV_ALGORITHMS = frozenset(('brute_force', 'brute_force_no_of_etas', 'brute_force_block_structure'))


def create_workflow(
    algorithm: Literal[tuple(IIV_ALGORITHMS)],
    iiv_strategy: Literal[tuple(IIV_STRATEGIES)] = 'no_add',
    rank_type: Literal[tuple(RANK_TYPES)] = 'mbic',
    cutoff: Optional[Union[float, int]] = None,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
    keep: Optional[List[str]] = None,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
):
    """Run IIVsearch tool. For more details, see :ref:`iivsearch`.

    Parameters
    ----------
    algorithm : {'brute_force', 'brute_force_no_of_etas', 'brute_force_block_structure'}
        Which algorithm to run.
    iiv_strategy : {'no_add', 'add_diagonal', 'fullblock', 'pd_add_diagonal', 'pd_fullblock'}
        If/how IIV should be added to start model. Default is 'no_add'.
    rank_type : {'ofv', 'lrt', 'aic', 'bic', 'mbic'}
        Which ranking type should be used. Default is mBIC.
    cutoff : float
        Cutoff for which value of the ranking function that is considered significant. Default
        is None (all models will be ranked)
    results : ModelfitResults
        Results for model
    model : Model
        Pharmpy model
    keep :  List[str]
        List of IIVs to keep
    strictness : str or None
        Strictness criteria

    Returns
    -------
    IIVSearchResults
        IIVsearch tool result object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> from pharmpy.tools import run_iivsearch, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> run_iivsearch('brute_force', results=results, model=model)   # doctest: +SKIP
    """

    wb = WorkflowBuilder(name='iivsearch')
    start_task = Task(
        'start_iiv',
        start,
        model,
        results,
        algorithm,
        iiv_strategy,
        rank_type,
        cutoff,
        keep,
        strictness,
    )
    wb.add_task(start_task)
    task_results = Task('results', _results)
    wb.add_task(task_results, predecessors=[start_task])
    return Workflow(wb)


def create_step_workflow(
    input_model_entry, base_model_entry, wf_algorithm, iiv_strategy, rank_type, cutoff, strictness
):
    wb = WorkflowBuilder()
    start_task = Task(f'start_{wf_algorithm.name}', _start_algorithm, base_model_entry)
    wb.add_task(start_task)

    if iiv_strategy != 'no_add':
        wf_fit = create_fit_workflow(n=1)
        wb.insert_workflow(wf_fit)
        base_model_task = wf_fit.output_tasks[0]
    else:
        base_model_task = start_task

    wb.insert_workflow(wf_algorithm)

    task_result = Task(
        'results',
        post_process,
        rank_type,
        cutoff,
        strictness,
        input_model_entry,
        base_model_entry.model.name,
    )

    post_process_tasks = [base_model_task] + wb.output_tasks
    wb.add_task(task_result, predecessors=post_process_tasks)

    return Workflow(wb)


def start(
    context, input_model, input_res, algorithm, iiv_strategy, rank_type, cutoff, keep, strictness
):
    input_model_entry = ModelEntry.create(input_model, modelfit_results=input_res)

    if iiv_strategy != 'no_add':
        base_model = update_initial_estimates(input_model, modelfit_results=input_res)
        base_model = _add_iiv(iiv_strategy, base_model, modelfit_results=input_res)
        base_model = base_model.replace(
            name='base_model', description=algorithms.create_description(base_model)
        )
        # FIXME: Set parent model once create_results fully supports model entries
        base_model_entry = ModelEntry.create(base_model, modelfit_results=None)
    else:
        base_model_entry = ModelEntry.create(input_model, modelfit_results=input_res)

    if algorithm == 'brute_force':
        list_of_algorithms = ['brute_force_no_of_etas', 'brute_force_block_structure']
    else:
        list_of_algorithms = [algorithm]
    sum_tools, sum_models, sum_inds, sum_inds_count, sum_errs = [], [], [], [], []

    no_of_models = 0
    last_res = None
    final_model_entry = None

    sum_models = [summarize_modelfit_results(input_res)]

    for algorithm_cur in list_of_algorithms:
        algorithm_func = getattr(algorithms, algorithm_cur)
        if algorithm_cur == "brute_force_no_of_etas":
            # NOTE: This does not need to be a model entry since it is only used as a start point for the
            # candidate models, when the workflow is run the input to this sub-workflow will be a model entry
            wf_algorithm = algorithm_func(
                base_model_entry.model, index_offset=no_of_models, keep=keep
            )
        else:
            wf_algorithm = algorithm_func(base_model_entry.model, index_offset=no_of_models)

        wf = create_step_workflow(
            input_model_entry,
            base_model_entry,
            wf_algorithm,
            iiv_strategy,
            rank_type,
            cutoff,
            strictness,
        )
        res = call_workflow(wf, f'results_{algorithm}', context)

        if base_model_entry.model.name in sum_models[-1].index.values:
            summary_models = res.summary_models.drop(base_model_entry.model.name, axis=0)
        else:
            summary_models = res.summary_models

        sum_tools.append(res.summary_tool)
        sum_models.append(summary_models)
        sum_inds.append(res.summary_individuals)
        sum_inds_count.append(res.summary_individuals_count)
        sum_errs.append(res.summary_errors)

        if res.final_model.name != input_model_entry.model.name:
            final_model_entry = context.model_database.retrieve_model_entry(res.final_model.name)
        else:
            final_model_entry = input_model_entry
        base_model_entry = final_model_entry
        iiv_strategy = 'no_add'
        last_res = res
        no_of_models = len(res.summary_tool) - 1

        assert base_model_entry is not None
        if (
            len(
                set(base_model_entry.model.random_variables.iiv.names).difference(
                    _get_fixed_etas(base_model_entry.model)
                )
            )
            <= 1
        ):
            break

    assert last_res is not None
    assert final_model_entry is not None

    input_model, input_res = input_model_entry.model, input_model_entry.modelfit_results
    final_model, final_res = final_model_entry.model, final_model_entry.modelfit_results

    # NOTE: Compute final final model
    final_final_model = last_res.final_model
    if input_res and final_res:
        bic_input = calculate_bic(input_model, input_res.ofv, type='iiv')
        bic_final = calculate_bic(final_model, final_res.ofv, type='iiv')
        if bic_final > bic_input:
            warnings.warn(
                f'Worse {rank_type} in final model {final_model.name} '
                f'({bic_final}) than {input_model.name} ({bic_input}), selecting '
                f'input model'
            )
            final_final_model = input_model

    keys = list(range(1, len(list_of_algorithms) + 1))

    return IIVSearchResults(
        summary_tool=_concat_summaries(sum_tools, keys),
        summary_models=_concat_summaries(sum_models, [0] + keys),  # To include input model
        summary_individuals=_concat_summaries(sum_inds, keys),
        summary_individuals_count=_concat_summaries(sum_inds_count, keys),
        summary_errors=_concat_summaries(sum_errs, keys),
        final_model=final_final_model,
        tool_database=last_res.tool_database,
    )


def _concat_summaries(summaries, keys):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA",
            category=FutureWarning,
        )
        return pd.concat(summaries, keys=keys, names=['step'])


def _results(res):
    return res


def _start_algorithm(model_entry):
    return model_entry


def _add_iiv(iiv_strategy, model, modelfit_results):
    assert iiv_strategy in ['add_diagonal', 'fullblock', 'pd_add_diagonal', 'pd_fullblock']
    if iiv_strategy in ['add_diagonal', 'fullblock']:
        model = add_pk_iiv(model)
        if iiv_strategy == 'fullblock':
            model = create_joint_distribution(
                model, individual_estimates=modelfit_results.individual_estimates
            )
    elif iiv_strategy in ['pd_add_diagonal', 'pd_fullblock']:
        model = add_pd_iiv(model)
        if iiv_strategy == 'pd_fullblock':
            model = create_joint_distribution(
                model, individual_estimates=modelfit_results.individual_estimates
            )
    return model


def post_process(rank_type, cutoff, strictness, input_model_entry, base_model_name, *model_entries):
    res_model_entries = []
    base_model_entry = None
    for model_entry in model_entries:
        if model_entry.model.name == base_model_name:
            base_model_entry = model_entry
        else:
            res_model_entries.append(model_entry)

    assert len(res_model_entries) > 0

    if not base_model_entry:
        raise ValueError('Error in workflow: No base model')

    # In order to have the IIV structure of the input model in the description column
    # in the result summaries
    if input_model_entry.model.name == base_model_entry.model.name:
        base_model = base_model_entry.model
        base_model_description = algorithms.create_description(base_model)
        base_model = base_model.replace(description=base_model_description)
        base_model_entry = ModelEntry.create(
            base_model, modelfit_results=base_model_entry.modelfit_results
        )

    res = create_results(
        IIVSearchResults,
        input_model_entry,
        base_model_entry,
        res_model_entries,
        rank_type,
        cutoff,
        bic_type='iiv',
        strictness=strictness,
    )

    summary_tool = res.summary_tool
    assert summary_tool is not None
    summary_models = summarize_modelfit_results(
        [model_entry.modelfit_results for model_entry in model_entries]
    )

    return replace(res, summary_models=summary_models)


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    algorithm,
    iiv_strategy,
    rank_type,
    model,
    keep,
    strictness,
):
    if keep:
        for parameter in keep:
            try:
                has_random_effect(model, parameter, "iiv")
            except KeyError:
                raise ValueError(f"Parameter {parameter} has no iiv.")

    if strictness is not None and "rse" in strictness.lower():
        if model.estimation_steps[-1].parameter_uncertainty_method is None:
            raise ValueError(
                'parameter_uncertainty_method not set for model, cannot calculate relative standard errors.'
            )


@dataclass(frozen=True)
class IIVSearchResults(ToolResults):
    pass
