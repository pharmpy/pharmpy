from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Optional, Union

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
from pharmpy.tools.common import (
    RANK_TYPES,
    ToolResults,
    create_plots,
    create_results,
    table_final_eta_shrinkage,
    update_initial_estimates,
)
from pharmpy.tools.iivsearch.algorithms import _get_fixed_etas, _remove_sublist
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import summarize_modelfit_results_from_entries
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder, call_workflow
from pharmpy.workflows.results import ModelfitResults

IIV_STRATEGIES = frozenset(
    ('no_add', 'add_diagonal', 'fullblock', 'pd_add_diagonal', 'pd_fullblock')
)
IIV_ALGORITHMS = frozenset(
    (
        'top_down_exhaustive',
        'bottom_up_stepwise',
        'skip',
    )
)
IIV_CORRELATION_ALGORITHMS = frozenset(
    (
        'top_down_exhaustive',
        'skip',
    )
)


def create_workflow(
    algorithm: Literal[tuple(IIV_ALGORITHMS)] = "top_down_exhaustive",
    iiv_strategy: Literal[tuple(IIV_STRATEGIES)] = 'no_add',
    rank_type: Literal[tuple(RANK_TYPES)] = 'mbic',
    cutoff: Optional[Union[float, int]] = None,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
    keep: Optional[Iterable[str]] = ("CL",),
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    correlation_algorithm: Optional[Literal[tuple(IIV_CORRELATION_ALGORITHMS)]] = None,
):
    """Run IIVsearch tool. For more details, see :ref:`iivsearch`.

    Parameters
    ----------
    algorithm : {'top_down_exhaustive','bottom_up_stepwise', 'skip'}
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
    keep : Iterable[str]
        List of IIVs to keep. Default is "CL"
    strictness : str or None
        Strictness criteria
    correlation_algorithm: {'top_down_exhaustive', 'skip'} or None
        Which algorithm to run for the determining block structure of added IIVs. If None, the
        algorithm is determined based on the 'algorithm' argument

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
    >>> run_iivsearch('td_brute_force', results=results, model=model)   # doctest: +SKIP
    """

    wb = WorkflowBuilder(name='iivsearch')
    start_task = Task(
        'start_iiv',
        start,
        model,
        results,
        algorithm,
        correlation_algorithm,
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
    input_model_entry,
    base_model_entry,
    wf_algorithm,
    iiv_strategy,
    rank_type,
    cutoff,
    strictness,
    context,
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
        wf_algorithm.name,
        context,
    )

    post_process_tasks = [base_model_task] + wb.output_tasks
    wb.add_task(task_result, predecessors=post_process_tasks)

    return Workflow(wb)


def start(
    context,
    input_model,
    input_res,
    algorithm,
    correlation_algorithm,
    iiv_strategy,
    rank_type,
    cutoff,
    keep,
    strictness,
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

    algorithm_sub = {
        "top_down_exhaustive": "td_exhaustive_no_of_etas",
        "bottom_up_stepwise": "bu_stepwise_no_of_etas",
    }
    correlation_algorithm_sub = {
        "top_down_exhaustive": "td_exhaustive_block_structure",
    }

    list_of_algorithms = []
    if algorithm != "skip":
        list_of_algorithms.append(algorithm_sub[algorithm])
    if correlation_algorithm != "skip":
        if correlation_algorithm is None:
            if algorithm in correlation_algorithm_sub.keys():
                correlation_algorithm = algorithm
            else:
                correlation_algorithm = "top_down_exhaustive"
        list_of_algorithms.append(correlation_algorithm_sub[correlation_algorithm])

    sum_tools, sum_models, sum_inds, sum_inds_count, sum_errs = [], [], [], [], []

    no_of_models = 0
    last_res = None
    final_model_entry = None

    sum_models = [summarize_modelfit_results_from_entries([input_model_entry])]

    applied_algorithms = []
    for algorithm_cur in list_of_algorithms:
        if (
            algorithm_cur == 'td_exhaustive_block_structure'
            and len(
                set(base_model_entry.model.random_variables.iiv.names).difference(
                    _get_fixed_etas(base_model_entry.model)
                )
            )
            <= 1
        ):
            continue
        algorithm_func = getattr(algorithms, algorithm_cur)
        if algorithm_cur == "td_exhaustive_no_of_etas":
            # NOTE: This does not need to be a model entry since it is only used as a start point for the
            # candidate models, when the workflow is run the input to this sub-workflow will be a model entry
            wf_algorithm = algorithm_func(
                base_model_entry.model, index_offset=no_of_models, keep=keep
            )
        elif algorithm_cur == "bu_stepwise_no_of_etas":
            wf_algorithm = algorithm_func(
                base_model_entry.model,
                strictness=strictness,
                index_offset=no_of_models,
                input_model_entry=input_model_entry,
                keep=keep,
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
            context,
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

        final_model = res.final_model
        if final_model.name != input_model_entry.model.name:
            final_model_entry = ModelEntry.create(
                model=final_model, modelfit_results=res.final_results
            )
        else:
            final_res = input_model_entry.modelfit_results
            final_model_entry = ModelEntry.create(model=final_model, modelfit_results=final_res)

        # FIXME: Add parent model
        base_model_entry = final_model_entry
        iiv_strategy = 'no_add'
        last_res = res
        no_of_models = len(res.summary_tool) - 1

        assert base_model_entry is not None

        applied_algorithms.append(algorithm_cur)

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

    keys = list(range(1, len(applied_algorithms) + 1))

    if final_final_model.name == final_model.name:
        final_results = final_res
    elif final_final_model.name == input_model.name:
        final_results = input_res

    plots = create_plots(final_final_model, final_results)

    return IIVSearchResults(
        summary_tool=_concat_summaries(sum_tools, keys),
        summary_models=_concat_summaries(sum_models, [0] + keys),  # To include input model
        summary_individuals=_concat_summaries(sum_inds, keys),
        summary_individuals_count=_concat_summaries(sum_inds_count, keys),
        summary_errors=_concat_summaries(sum_errs, keys),
        final_model=final_final_model,
        final_results=final_results,
        final_model_dv_vs_ipred_plot=plots['dv_vs_ipred'],
        final_model_dv_vs_pred_plot=plots['dv_vs_pred'],
        final_model_cwres_vs_idv_plot=plots['cwres_vs_idv'],
        final_model_abs_cwres_vs_ipred_plot=plots['abs_cwres_vs_ipred'],
        final_model_eta_distribution_plot=plots['eta_distribution'],
        final_model_eta_shrinkage=table_final_eta_shrinkage(final_final_model, final_results),
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


def post_process(
    rank_type,
    cutoff,
    strictness,
    input_model_entry,
    base_model_name,
    algorithm_name,
    context,
    *model_entries,
):
    res_model_entries = []
    base_model_entry = None

    def flatten_list(lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(flatten_list(item))
            else:
                result.append(item)
        return result

    model_entries = flatten_list(model_entries)
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

    # Uses other values than default for MBIC calculations
    if rank_type == "mbic" and algorithm_name == "bu_stepwise_no_of_etas":
        # Find all ETAs in model
        iivs = base_model_entry.model.random_variables.iiv
        iiv_names = iivs.names  # All ETAs in the base model
        # Remove fixed etas
        fixed_etas = _get_fixed_etas(base_model_entry.model)
        iiv_names = _remove_sublist(iiv_names, fixed_etas)

        number_of_predicted = len(iiv_names)
        number_of_expected = number_of_predicted / 2
    else:
        number_of_predicted = None
        number_of_expected = None

    res = create_results(
        IIVSearchResults,
        input_model_entry,
        base_model_entry,
        res_model_entries,
        rank_type,
        cutoff,
        bic_type='iiv',
        strictness=strictness,
        n_predicted=number_of_predicted,
        n_expected=number_of_expected,
        context=context,
    )

    summary_tool = res.summary_tool
    assert summary_tool is not None
    summary_models = summarize_modelfit_results_from_entries(model_entries)

    return replace(res, summary_models=summary_models)


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    algorithm, iiv_strategy, rank_type, model, keep, strictness, correlation_algorithm
):
    if keep and model:
        for parameter in keep:
            try:
                has_random_effect(model, parameter, "iiv")
            except KeyError:
                warnings.warn(f"Parameter {keep} has no iiv and is ignored")

    if strictness is not None and "rse" in strictness.lower():
        if model.execution_steps[-1].parameter_uncertainty_method is None:
            raise ValueError(
                'parameter_uncertainty_method not set for model, cannot calculate relative standard errors.'
            )

    if algorithm == correlation_algorithm == "skip":
        raise ValueError("Both algorithm and correlation_algorithm are set to 'skip'")
    elif algorithm == "skip" and correlation_algorithm is None:
        raise ValueError(
            "correlation_algorithm need to be specified if" " 'algorithm' is set to skip"
        )


@dataclass(frozen=True)
class IIVSearchResults(ToolResults):
    rst_path = Path(__file__).resolve().parent / 'report.rst'
    pass
