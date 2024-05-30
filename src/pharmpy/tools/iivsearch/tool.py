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
    create_joint_distribution,
    find_clearance_parameters,
    has_random_effect,
)
from pharmpy.tools.common import (
    RANK_TYPES,
    ToolResults,
    create_plots,
    create_results,
    summarize_tool,
    table_final_eta_shrinkage,
    update_initial_estimates,
)
from pharmpy.tools.iivsearch.algorithms import _get_fixed_etas
from pharmpy.tools.linearize.delinearize import delinearize_model
from pharmpy.tools.linearize.tool import create_workflow as create_linearize_workflow
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import calculate_bic_penalty, summarize_modelfit_results_from_entries
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
    rank_type: Literal[tuple(RANK_TYPES)] = 'bic',
    linearize: bool = False,
    cutoff: Optional[Union[float, int]] = None,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
    keep: Optional[Iterable[str]] = ("CL",),
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    correlation_algorithm: Optional[Literal[tuple(IIV_CORRELATION_ALGORITHMS)]] = None,
    E_p: Optional[float] = None,
    E_q: Optional[float] = None,
):
    """Run IIVsearch tool. For more details, see :ref:`iivsearch`.

    Parameters
    ----------
    algorithm : {'top_down_exhaustive','bottom_up_stepwise', 'skip'}
        Which algorithm to run when determining number of IIVs.
    iiv_strategy : {'no_add', 'add_diagonal', 'fullblock', 'pd_add_diagonal', 'pd_fullblock'}
        If/how IIV should be added to start model. Default is 'no_add'.
    rank_type : {'ofv', 'lrt', 'aic', 'bic', 'mbic'}
        Which ranking type should be used. Default is BIC.
    linearize : bool
        Wheter or not use linearization when running the tool.
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
    E_p : float
        Expected number of predictors for diagonal elements (used for mBIC). Must be set when using mBIC and
        when the argument 'algorithm' is not 'skip'
    E_q : float
        Expected number of predictors for off-diagonal elements (used for mBIC). Must be set when using mBIC
        and when the argument `correlation_algorithm` is not `skip` or None

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
        E_p,
        E_q,
        linearize,
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
    E_p,
    E_q,
    cutoff,
    strictness,
    list_of_algorithms,
    ref_model,
    keep,
    linearize,
    param_mapping,
    context,
):
    wb = WorkflowBuilder()
    start_task = Task(f'start_{wf_algorithm.name}', _start_algorithm, base_model_entry)
    wb.add_task(start_task)

    if (wf_algorithm.name == 'td_exhaustive_no_of_etas' and iiv_strategy != 'no_add') or (
        wf_algorithm.name == 'bu_stepwise_no_of_etas' and iiv_strategy != 'no_add' and not linearize
    ):
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
        list_of_algorithms,
        ref_model,
        E_p,
        E_q,
        keep,
        linearize,
        param_mapping,
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
    E_p,
    E_q,
    linearize,
    cutoff,
    keep,
    strictness,
):
    input_model = input_model.replace(
        name="input", description=algorithms.create_description(input_model)
    )
    input_model_entry = ModelEntry.create(input_model, modelfit_results=input_res)
    context.store_input_model_entry(input_model_entry)

    if iiv_strategy != 'no_add':
        base_model = update_initial_estimates(input_model, modelfit_results=input_res)
        base_model = _add_iiv(iiv_strategy, base_model, modelfit_results=input_res)
        base_model = base_model.replace(
            name='base', description=algorithms.create_description(base_model)
        )
        # FIXME: Set parent model once create_results can do different things for different tools
        base_model_entry = ModelEntry.create(base_model, modelfit_results=None)
    else:
        base_model = input_model.replace(name='base')
        base_model_entry = ModelEntry.create(base_model, modelfit_results=input_res)

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

    # LINEARIZE
    if linearize:
        # Create param map for ETA
        from .algorithms import _create_param_dict

        param_mapping = _create_param_dict(
            base_model_entry.model, dists=base_model_entry.model.random_variables.iiv
        )

        linearize_context = context.create_subcontext('linearization')
        linear_workflow = create_linearize_workflow(
            model=base_model_entry.model,
            model_name="linear_base_model",
            description=algorithms.create_description(base_model_entry.model),
        )
        linear_results = call_workflow(linear_workflow, "running_linearization", linearize_context)
        base_model_entry = ModelEntry.create(
            model=linear_results.final_model, modelfit_results=linear_results.final_model_results
        )
    else:
        param_mapping = None

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
                base_model_entry.model,
                index_offset=no_of_models,
                keep=keep,
                param_mapping=param_mapping,
            )
        elif algorithm_cur == "bu_stepwise_no_of_etas":
            try:
                clearance_parameter = find_clearance_parameters(input_model)
            except ValueError:
                pass
            if clearance_parameter:
                clearance_parameter = str(clearance_parameter[0])
            else:
                clearance_parameter = ""
            wf_algorithm = algorithm_func(
                base_model_entry.model,
                strictness=strictness,
                index_offset=no_of_models,
                input_model_entry=input_model_entry,
                list_of_algorithms=list_of_algorithms,
                rank_type=rank_type,
                keep=keep,
                param_mapping=param_mapping,
                clearance_parameter=clearance_parameter,
            )
        else:
            wf_algorithm = algorithm_func(
                base_model_entry.model, index_offset=no_of_models, param_mapping=param_mapping
            )

        wf = create_step_workflow(
            input_model_entry,
            base_model_entry,
            wf_algorithm,
            iiv_strategy=iiv_strategy,
            rank_type=rank_type,
            E_p=E_p,
            E_q=E_q,
            cutoff=cutoff,
            strictness=strictness,
            list_of_algorithms=list_of_algorithms,
            ref_model=base_model,
            keep=keep,
            linearize=linearize,
            param_mapping=param_mapping,
            context=context,
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
        if rank_type == 'mbic':
            penalties = _get_penalties(
                base_model,
                [input_model_entry, final_model_entry],
                keep=keep,
                list_of_algorithms=list_of_algorithms,
                E_p=E_p,
                E_q=E_q,
            )
        else:
            penalties = None
        summary_final_step = summarize_tool(
            [final_model_entry],
            input_model_entry,
            rank_type,
            cutoff=cutoff,
            bic_type='iiv',
            strictness=strictness,
            penalties=penalties,
        )
        sum_tools.append(summary_final_step)
        best_model_name = summary_final_step['rank'].idxmin()

        if best_model_name == input_model.name:
            warnings.warn(
                f'Worse {rank_type} in final model {final_model.name} '
                f'than {input_model.name}, selecting input model'
            )
            final_final_model = input_model

    keys = list(range(1, len(applied_algorithms) + 1))

    if final_final_model.name == final_model.name:
        final_results = final_res
    elif final_final_model.name == input_model.name:
        final_results = input_res

    plots = create_plots(final_final_model, final_results)

    final_final_model = final_final_model.replace(name="final")
    context.store_final_model_entry(final_final_model)

    final_results = IIVSearchResults(
        summary_tool=_concat_summaries(
            sum_tools, keys + [len(keys) + 1]
        ),  # To include step comparing input to final
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

    return final_results


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


def rename_linbase(model, linbase_model_entry):
    linbase_model = linbase_model_entry.replace(
        name="linear_base_model", description=algorithms.create_description(model)
    )
    return ModelEntry.create(model=linbase_model)


def _add_iiv(iiv_strategy, model, modelfit_results):
    # IF LINEARIZED - inital value should be 0.00001
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
    list_of_algorithms,
    ref_model,
    E_p,
    E_q,
    keep,
    linearize,
    param_mapping,
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

    if rank_type == "mbic":
        penalties = _get_penalties(
            ref_model, model_entries, keep, list_of_algorithms, E_p=E_p, E_q=E_q
        )
    else:
        penalties = None

    res = create_results(
        IIVSearchResults,
        input_model_entry,
        base_model_entry,
        res_model_entries,
        rank_type=rank_type,
        cutoff=cutoff,
        bic_type='iiv',
        strictness=strictness,
        penalties=penalties,
        context=context,
    )

    if linearize:
        final_linearized_model = res.final_model
        flm_etas = final_linearized_model.random_variables.iiv.names
        final_param_mapp = {k: v for k, v in param_mapping.items() if k in flm_etas}
        final_delinearized_model = delinearize_model(
            final_linearized_model, input_model_entry.model, final_param_mapp
        )
        final_delinearized_model = final_delinearized_model.replace(
            name=f'delinerized_{final_delinearized_model.name}',
            description=algorithms.create_description(final_delinearized_model),
        )

        lin_model_entry = ModelEntry.create(
            model=final_delinearized_model, parent=final_linearized_model
        )
        dl_wf = WorkflowBuilder(name="delinearization_workflow")
        l_start = Task("START", _start_algorithm, lin_model_entry)
        dl_wf.add_task(l_start)
        fit_wf = create_fit_workflow(n=1)
        dl_wf.insert_workflow(fit_wf)
        dlin_model_entry = call_workflow(Workflow(dl_wf), "running_delinearization", context)

        res_model_entries.append(dlin_model_entry)
        res = create_results(
            IIVSearchResults,
            input_model_entry,
            base_model_entry,
            res_model_entries,
            rank_type,
            cutoff,
            bic_type='iiv',
            strictness=strictness,
            context=context,
        )

        res = replace(res, final_model=dlin_model_entry.model)
        summary_tool = res.summary_tool
        assert summary_tool is not None
        summary_tool = modify_summary_tool(res.summary_tool, dlin_model_entry.model.name)
        res = replace(res, summary_tool=summary_tool)
        summary_models = summarize_modelfit_results_from_entries(model_entries)
    else:
        summary_tool = res.summary_tool
        assert summary_tool is not None
        summary_models = summarize_modelfit_results_from_entries(model_entries)

    return replace(res, summary_models=summary_models)


def _get_penalties(ref_model, candidate_model_entries, keep, list_of_algorithms, E_p, E_q):
    search_space = []
    if any('no_of_etas' in algorithm for algorithm in list_of_algorithms):
        search_space.append('iiv_diag')
    if any('block' in algorithm for algorithm in list_of_algorithms):
        search_space.append('iiv_block')
    penalties = [
        calculate_bic_penalty(
            me.model, search_space, base_model=ref_model, keep=keep, E_p=E_p, E_q=E_q
        )
        for me in candidate_model_entries
    ]
    return penalties


def modify_summary_tool(summary_tool, first_model_name):
    # If linear model --> Force de-linearized model to be chosen
    # TODO : Remove BIC values of linearized models as they are misleading ?
    summary_tool = summary_tool.reset_index()
    first_model_entry_rank = summary_tool.loc[summary_tool["model"] == first_model_name][
        "rank"
    ].iloc[0]
    summary_tool.loc[summary_tool['rank'] < first_model_entry_rank, 'rank'] += 1
    summary_tool.loc[summary_tool['model'] == first_model_name, 'rank'] = 1
    summary_tool = summary_tool.sort_values(by=['rank'], ascending=True)
    summary_tool = summary_tool.set_index(['model'])
    return summary_tool


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    algorithm, iiv_strategy, rank_type, model, keep, strictness, correlation_algorithm, E_p, E_q
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

    if rank_type != 'mbic' and (E_p is not None or E_q is not None):
        raise ValueError(
            f'E_p and E_q can only be provided when `rank_type` is mbic: got `{rank_type}`'
        )
    if rank_type == 'mbic':
        if algorithm != 'skip' and E_p is None:
            raise ValueError('Value `E_p` must be provided for `algorithm` when using mbic')
        if correlation_algorithm and correlation_algorithm != 'skip' and E_q is None:
            raise ValueError(
                'Value `E_q` must be provided for `correlation_algorithm` when using mbic'
            )
        if E_p is not None and E_p <= 0.0:
            raise ValueError(f'Value `E_p` must be more than 0: got `{E_p}`')
        if E_q is not None and E_q <= 0.0:
            raise ValueError(f'Value `E_q` must be more than 0: got `{E_q}`')


@dataclass(frozen=True)
class IIVSearchResults(ToolResults):
    rst_path = Path(__file__).resolve().parent / 'report.rst'
    pass
