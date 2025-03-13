from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
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
    fix_parameters,
    has_random_effect,
    unfix_parameters,
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
from pharmpy.tools.iivsearch.algorithms import _get_fixed_etas, get_eta_names
from pharmpy.tools.linearize.delinearize import delinearize_model
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import calculate_mbic_penalty, summarize_modelfit_results_from_entries
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
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
    model: Model,
    results: ModelfitResults,
    algorithm: Literal[tuple(IIV_ALGORITHMS)] = "top_down_exhaustive",
    iiv_strategy: Literal[tuple(IIV_STRATEGIES)] = 'no_add',
    rank_type: Literal[tuple(RANK_TYPES)] = 'bic',
    linearize: bool = False,
    cutoff: Optional[Union[float, int]] = None,
    keep: Optional[Iterable[str]] = ("CL",),
    strictness: str = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    correlation_algorithm: Optional[Literal[tuple(IIV_CORRELATION_ALGORITHMS)]] = None,
    E_p: Optional[Union[float, str]] = None,
    E_q: Optional[Union[float, str]] = None,
):
    """Run IIVsearch tool. For more details, see :ref:`iivsearch`.

    Parameters
    ----------
    model : Model
        Pharmpy model
    results : ModelfitResults
        Results for model
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
    >>> run_iivsearch(model=model, results=results, algorithm='td_brute_force')   # doctest: +SKIP
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
    keep,
    context,
):
    wb = WorkflowBuilder()
    start_task = Task(f'start_{wf_algorithm.name}', _start_algorithm, base_model_entry)
    wb.add_task(start_task)

    if wf_algorithm.name == 'td_exhaustive_no_of_etas' and iiv_strategy != 'no_add':
        wf_fit = create_fit_workflow(n=1)
        wb.insert_workflow(wf_fit)
        base_model_task = [wf_fit.output_tasks[0]]
    elif wf_algorithm.name == 'bu_stepwise_no_of_etas':
        base_model_task = []
    else:
        base_model_task = [start_task]

    wb.insert_workflow(wf_algorithm)

    task_result = Task(
        'results',
        post_process,
        rank_type,
        cutoff,
        strictness,
        input_model_entry,
        wf_algorithm.name,
        E_p,
        E_q,
        keep,
        context,
    )

    post_process_tasks = base_model_task + wb.output_tasks
    wb.add_task(task_result, predecessors=post_process_tasks)

    return Workflow(wb)


def prepare_input_model(input_model, input_res):
    input_model = input_model.replace(
        name="input", description=algorithms.create_description(input_model)
    )
    input_model_entry = ModelEntry.create(input_model, modelfit_results=input_res)
    return input_model, input_model_entry


def prepare_base_model(input_model_entry, iiv_strategy, linearize):
    if iiv_strategy != 'no_add':
        base_model = update_initial_estimates(
            input_model_entry.model,
            modelfit_results=input_model_entry.modelfit_results,
            move_est_close_to_bounds=not linearize,
        )
        base_model = add_iiv(
            iiv_strategy,
            base_model,
            modelfit_results=input_model_entry.modelfit_results,
            linearize=linearize,
        )
        # FIXME: Set parent model once create_results can do different things for different tools
        base_model = base_model.replace(name='base')
        mfr = None
    else:
        base_model = input_model_entry.model
        mfr = input_model_entry.modelfit_results
    base_model = base_model.replace(description=algorithms.create_description(base_model))
    base_model_entry = ModelEntry.create(base_model, modelfit_results=mfr)
    return base_model, base_model_entry


def prepare_algorithms(algorithm, correlation_algorithm):
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
    return list_of_algorithms


def create_param_mapping(me, linearize):
    if linearize:
        from .algorithms import _create_param_dict

        param_mapping = _create_param_dict(me.model, dists=me.model.random_variables.iiv)
    else:
        param_mapping = None
    return param_mapping


def run_linearization(context, baseme):
    from pharmpy.tools.run import run_subtool

    linear_results = run_subtool(
        'linearize',
        context,
        model=baseme.model,
        description=algorithms.create_description(baseme.model),
    )
    linbaseme = ModelEntry.create(
        model=linear_results.final_model, modelfit_results=linear_results.final_model_results
    )
    return linbaseme


def update_linearized_base_model(baseme, input_model, iiv_strategy, param_mapping):
    if iiv_strategy == 'no_add':
        return baseme
    added_params = baseme.model.parameters - input_model.parameters
    model = unfix_parameters(baseme.model, added_params.names)
    if iiv_strategy in ('fullblock', 'pd_fullblock'):
        model = create_joint_distribution(
            model, individual_estimates=baseme.modelfit_results.individual_estimates
        )
    descr = algorithms.create_description(model, iov=False, param_dict=param_mapping)
    model = model.replace(name="base", description=descr)
    return ModelEntry.create(model=model, modelfit_results=None)


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
    context.log_info("Starting tool iivsearch")
    input_model, input_model_entry = prepare_input_model(input_model, input_res)
    context.store_input_model_entry(input_model_entry)
    context.log_info(f"Input model OFV: {input_res.ofv:.3f}")

    list_of_algorithms = prepare_algorithms(algorithm, correlation_algorithm)

    sum_tools, sum_errs = [], []
    no_of_models = 0
    last_res = None
    final_model_entry = None
    sum_models = [summarize_modelfit_results_from_entries([input_model_entry])]

    if algorithm != 'no_add':
        context.log_info("Creating base model")
    base_model, base_model_entry = prepare_base_model(input_model_entry, iiv_strategy, linearize)

    param_mapping = create_param_mapping(base_model_entry, linearize)

    if linearize:
        base_model_entry = run_linearization(context, base_model_entry)
        base_model_entry = update_linearized_base_model(
            base_model_entry, input_model, iiv_strategy, param_mapping
        )

    applied_algorithms = []
    for i, algorithm_cur in enumerate(list_of_algorithms, start=1):
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
                E_p=E_p,
                E_q=E_q,
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
            keep=keep,
            context=context,
        )
        context.log_info(f"Starting step '{algorithm_cur}'")
        res = context.call_workflow(wf, f'results_{algorithm}')

        if wf_algorithm.name == 'bu_stepwise_no_of_etas':
            ref_model_name = 'iivsearch_run1'
        else:
            ref_model_name = base_model_entry.model.name

        if ref_model_name in sum_models[-1].index.values:
            summary_models = res.summary_models.drop(base_model_entry.model.name, axis=0)
        else:
            summary_models = res.summary_models

        sum_tools.append(res.summary_tool)
        sum_models.append(summary_models)
        sum_errs.append(res.summary_errors)

        final_model = res.final_model
        if final_model.name != input_model_entry.model.name:
            final_model_entry = ModelEntry.create(
                model=final_model, modelfit_results=res.final_results
            )
        else:
            final_res = input_model_entry.modelfit_results
            final_model_entry = ModelEntry.create(model=final_model, modelfit_results=final_res)
        descr = final_model_entry.model.description
        ofv = final_model_entry.modelfit_results.ofv
        context.log_info(f"Finished step '{algorithm_cur}'. Best model: {descr}, OFV: {ofv:.3f}")

        # FIXME: Add parent model
        base_model_entry = final_model_entry
        iiv_strategy = 'no_add'
        last_res = res
        no_of_models = len(res.summary_tool) - 1
        if wf_algorithm.name == 'bu_stepwise_no_of_etas':
            no_of_models += 1

        assert base_model_entry is not None

        applied_algorithms.append(algorithm_cur)

    assert last_res is not None
    assert final_model_entry is not None

    if linearize:
        final_linearized_model = final_model_entry.model
        dl_wf = create_delinearize_workflow(
            input_model_entry.model, final_linearized_model, param_mapping, i
        )
        context.log_info('Running delinearized model')
        dlin_model_entry = context.call_workflow(Workflow(dl_wf), "running_delinearization")
        try:
            sum_tool = summarize_tool(
                [dlin_model_entry],
                dlin_model_entry,
                rank_type=rank_type,
                cutoff=cutoff,
                bic_type='iiv',
                strictness=strictness,
                penalties=None,
            )
        except ValueError:
            context.abort_workflow('Delinearized model failed strictness criteria')
        sum_model = summarize_modelfit_results_from_entries([dlin_model_entry])
        last_res = IIVSearchResults(
            summary_tool=sum_tool,
            summary_models=sum_model,
            final_model=dlin_model_entry.model,
            final_results=dlin_model_entry.modelfit_results,
        )

        sum_tools.append(sum_tool)
        sum_models.append(summarize_modelfit_results_from_entries([dlin_model_entry]))
        final_model_entry = dlin_model_entry

    input_model, input_res = input_model_entry.model, input_model_entry.modelfit_results
    final_model, final_res = final_model_entry.model, final_model_entry.modelfit_results

    # NOTE: Compute final final model
    final_final_model = last_res.final_model
    if input_res and final_res:
        if rank_type == 'mbic':
            penalties = get_mbic_penalties(
                base_model,
                [input_model, final_model],
                keep=keep,
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
            context.log_warning(
                f'Worse {rank_type} in final model {final_model.name} '
                f'than {input_model.name}, selecting input model'
            )
            final_final_model = input_model

    if final_final_model.name == final_model.name:
        final_results = final_res
    elif final_final_model.name == input_model.name:
        final_results = input_res

    plots = create_plots(final_final_model, final_results)

    context.store_final_model_entry(final_final_model)

    keys = list(range(1, len(applied_algorithms) + 1))
    keys_summary_tool = keys + [len(keys) + 1]  # Include step comparing input to final
    keys_summary_models = [0] + keys  # Include input model
    if linearize:
        keys_summary_tool += [len(keys) + 2]
        keys_summary_models += [len(keys) + 1]

    final_results = IIVSearchResults(
        summary_tool=_concat_summaries(
            sum_tools, keys_summary_tool
        ),  # To include step comparing input to final
        summary_models=_concat_summaries(sum_models, keys_summary_models),  # To include input model
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


def get_ref_model(models, algorithm):
    def _no_of_params(model):
        return len(model.random_variables.iiv.parameter_names)

    if algorithm.startswith('td'):
        return max(models, key=_no_of_params)
    elif algorithm.startswith('bu'):
        return min(models, key=_no_of_params)
    else:
        raise ValueError(f'Unknown ref model type: {algorithm}')


def _concat_summaries(summaries, keys):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA",
            category=FutureWarning,
        )
        return pd.concat(summaries, keys=keys, names=['step'])


def _results(context, res):
    context.log_info("Finishing tool iivsearch")
    return res


def _start_algorithm(model_entry):
    return model_entry


def add_iiv(iiv_strategy, model, modelfit_results, linearize=False):
    assert iiv_strategy in ('add_diagonal', 'fullblock', 'pd_add_diagonal', 'pd_fullblock')

    if linearize:
        init = 0.000001
    else:
        init = 0.09

    if iiv_strategy in ('add_diagonal', 'fullblock'):
        new = add_pk_iiv(model, initial_estimate=init)
    elif iiv_strategy in ('pd_add_diagonal', 'pd_fullblock'):
        new = add_pd_iiv(model, initial_estimate=init)
    if linearize:
        added_params = new.parameters - model.parameters
        new = fix_parameters(new, added_params.names)
    elif iiv_strategy in ('fullblock', 'pd_fullblock'):
        # To exclude e.g. IIV on RUV
        eta_names = get_eta_names(new, [], {})
        new = create_joint_distribution(
            new, eta_names, individual_estimates=modelfit_results.individual_estimates
        )
    return new


def post_process(
    rank_type,
    cutoff,
    strictness,
    input_model_entry,
    algorithm,
    E_p,
    E_q,
    keep,
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

    base_model = get_ref_model([me.model for me in model_entries], algorithm)
    for model_entry in model_entries:
        if model_entry.model.name == base_model.name:
            base_model_entry = model_entry
        else:
            res_model_entries.append(model_entry)

    if algorithm == 'bu_stepwise_no_of_etas':
        base_model_entry = ModelEntry.create(
            base_model_entry.model,
            modelfit_results=base_model_entry.modelfit_results,
            parent=input_model_entry.model,
        )

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
        models = [me.model for me in model_entries]
        if algorithm == 'bu_stepwise_no_of_etas':
            ref_model = get_ref_model(models, 'td')
        else:
            ref_model = base_model
        penalties = get_mbic_penalties(ref_model, models, keep, E_p=E_p, E_q=E_q)
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

    summary_tool = res.summary_tool
    assert summary_tool is not None
    summary_models = summarize_modelfit_results_from_entries(model_entries)

    return replace(res, summary_models=summary_models)


def create_delinearize_workflow(input_model, final_model, param_mapping, stepno):
    flm_etas = final_model.random_variables.iiv.names
    final_param_map = {k: v for k, v in param_mapping.items() if k in flm_etas}
    final_delinearized_model = delinearize_model(final_model, input_model, final_param_map)
    final_delinearized_model = final_delinearized_model.replace(
        name=f'delinearized{stepno}',
        description=algorithms.create_description(final_delinearized_model),
    )

    lin_model_entry = ModelEntry.create(
        model=final_delinearized_model,
    )
    dl_wf = WorkflowBuilder(name="delinearization_workflow")
    l_start = Task("START", _start_algorithm, lin_model_entry)
    dl_wf.add_task(l_start)
    fit_wf = create_fit_workflow(n=1)
    dl_wf.insert_workflow(fit_wf)

    return dl_wf


def get_mbic_penalties(ref_model, candidate_models, keep, E_p, E_q):
    search_space = []
    if E_p:
        search_space.append('iiv_diag')
    if E_q:
        search_space.append('iiv_block')
    penalties = [
        calculate_mbic_penalty(
            model, search_space, base_model=ref_model, keep=keep, E_p=E_p, E_q=E_q
        )
        for model in candidate_models
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
            if parameter not in map(lambda x: str(x), model.statements.free_symbols):
                raise ValueError(f'Symbol `{parameter}` does not exist in input model')
            if not has_random_effect(model, parameter, "iiv"):
                warnings.warn(f"Parameter `{parameter}` has no iiv and is ignored")

    if strictness is not None and "rse" in strictness.lower():
        if model.execution_steps[-1].parameter_uncertainty_method is None:
            raise ValueError(
                '`parameter_uncertainty_method` not set for model, cannot calculate relative standard errors.'
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
        if isinstance(E_p, float) and E_p <= 0.0:
            raise ValueError(f'Value `E_p` must be more than 0: got `{E_p}`')
        if isinstance(E_q, float) and E_q <= 0.0:
            raise ValueError(f'Value `E_q` must be more than 0: got `{E_q}`')
        if isinstance(E_p, str) and not E_p.endswith('%'):
            raise ValueError(f'Value `E_p` must be denoted with `%`: got `{E_p}`')
        if isinstance(E_q, str) and not E_q.endswith('%'):
            raise ValueError(f'Value `E_q` must be denoted with `%`: got `{E_q}`')


@dataclass(frozen=True)
class IIVSearchResults(ToolResults):
    pass
