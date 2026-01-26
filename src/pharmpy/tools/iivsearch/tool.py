from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import pharmpy.tools.iivsearch.algorithms as algorithms
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.mfl import ModelFeatures
from pharmpy.model import Model
from pharmpy.modeling import (
    create_joint_distribution,
    fix_parameters,
    get_rv_parameters,
    set_initial_estimates,
    unfix_parameters,
)
from pharmpy.modeling.mfl import (
    expand_model_features,
    get_model_features,
    is_in_search_space,
    transform_into_search_space,
)
from pharmpy.tools.common import (
    RANK_TYPES,
    ToolResults,
    add_parent_column,
    concat_summaries,
    create_plots,
    table_final_eta_shrinkage,
    update_initial_estimates,
)
from pharmpy.tools.linearize.delinearize import delinearize_model
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import (
    run_subtool,
    summarize_modelfit_results,
)
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults

from .algorithms import create_description, get_best_model_entry, rank_models

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


@dataclass(frozen=True)
class RankingOptions:
    rank_type: str
    cutoff: Optional[float]
    strictness: str
    parameter_uncertainty_method: str
    E: Optional[tuple[Union[float, str], Union[float, str]]]
    search_space: Optional[str]


def create_workflow(
    model: Model,
    results: ModelfitResults,
    algorithm: Literal[tuple(IIV_ALGORITHMS)] = "top_down_exhaustive",
    search_space: Optional[str] = None,
    as_fullblock: bool = False,
    rank_type: Literal[tuple(RANK_TYPES)] = 'bic',
    linearize: bool = False,
    cutoff: Optional[Union[float, int]] = None,
    strictness: str = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    correlation_algorithm: Optional[Literal[tuple(IIV_CORRELATION_ALGORITHMS)]] = None,
    E_p: Optional[Union[float, str]] = None,
    E_q: Optional[Union[float, str]] = None,
    parameter_uncertainty_method: Optional[Literal['SANDWICH', 'SMAT', 'RMAT', 'EFIM']] = None,
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
    search_space : str
        Search space to explore
    as_fullblock : bool
        Whether to add IIVs as a fullblock
    rank_type : {'ofv', 'lrt', 'aic', 'bic', 'mbic'}
        Which ranking type should be used. Default is BIC.
    linearize : bool
        Wheter or not use linearization when running the tool.
    cutoff : float
        Cutoff for which value of the ranking function that is considered significant. Default
        is None (all models will be ranked)
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
    parameter_uncertainty_method : {'SANDWICH', 'SMAT', 'RMAT', 'EFIM'} or None
        Parameter uncertainty method. Will be used in ranking models if strictness includes
        parameter uncertainty

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
    >>> run_iivsearch(model=model, results=results, algorithm='top_down_exhaustive')   # doctest: +SKIP
    """
    wb = WorkflowBuilder(name='iivsearch')

    mfl = ModelFeatures.create(search_space)

    rank_options = prepare_rank_options(
        rank_type,
        cutoff,
        strictness,
        parameter_uncertainty_method,
        E_p,
        E_q,
        expand_model_features(model, mfl),
    )

    steps_to_run = prepare_algorithms(algorithm, correlation_algorithm)

    start_task = Task.create('start', start, model, results)
    wb.add_task(start_task)

    if not linearize:
        wb, search_tasks = insert_search_workflow(
            wb, model, steps_to_run, mfl, as_fullblock, rank_options
        )
    else:
        wb, search_tasks = insert_linearized_search_workflow(
            wb, model, steps_to_run, mfl, as_fullblock, rank_options
        )

    compare_task = Task.create(
        'compare_to_input',
        compare_to_input_model,
        rank_options,
    )
    wb.add_task(compare_task, predecessors=[start_task, search_tasks[-1]])

    unpack_task = Task.create('unpack', unpack_tool_summaries)
    wb.add_task(unpack_task)

    for i, search_task in enumerate(search_tasks):
        if i == len(search_tasks) - 1:
            break
        dest = (search_tasks[i + 1], unpack_task)
        wb.scatter(search_task, dest)

    post_process_task = Task.create('postprocess', postprocess)
    wb.add_task(post_process_task, predecessors=[compare_task, unpack_task])

    results_task = Task('results', _results)
    wb.add_task(results_task, predecessors=[post_process_task])

    return Workflow(wb)


def prepare_rank_options(
    rank_type, cutoff, strictness, parameter_uncertainty_method, E_p, E_q, search_space
):
    assert search_space.is_expanded()

    E = (E_p, E_q) if E_p is not None or E_q is not None else None
    search_space = repr(search_space) if rank_type == 'mbic' else None

    rank_type = rank_type + '_iiv' if rank_type in ('bic', 'mbic') else rank_type

    rank_options = RankingOptions(
        rank_type=rank_type,
        cutoff=cutoff,
        strictness=strictness,
        parameter_uncertainty_method=parameter_uncertainty_method,
        E=E,
        search_space=search_space,
    )
    return rank_options


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


def start(context, input_model, input_res):
    context.log_info("Starting tool iivsearch")
    input_model_entry = prepare_input_model_entry(input_model, input_res)
    context.store_input_model_entry(input_model_entry)
    context.log_info(f"Input model OFV: {input_res.ofv:.3f}")
    return input_model_entry


def prepare_input_model_entry(input_model, input_res):
    mfl = get_model_features(input_model)
    description = create_description(mfl, type='iiv')
    input_model = input_model.replace(name="input", description=description)
    input_model_entry = ModelEntry.create(input_model, modelfit_results=input_res)
    return input_model_entry


def insert_search_workflow(wb, model, steps_to_run, mfl, as_fullblock, rank_options):
    mfl_expanded = expand_model_features(model, mfl)
    base_type = 'td' if steps_to_run[0].startswith('td') else 'bu'
    if needs_base_model(model, mfl_expanded, as_fullblock, base_type):
        create_base_task = Task.create(
            'base_model', _create_base_model_entry, base_type, mfl_expanded, as_fullblock
        )
        wb.add_task(create_base_task, predecessors=wb.output_tasks[0])
        wf_fit = create_fit_workflow(n=1)
        wb.insert_workflow(wf_fit)

    init_search_task = Task.create('init_search', init_search)
    wb.add_task(init_search_task, predecessors=wb.output_tasks[0])

    search_tasks = []
    for step in steps_to_run:
        if 'exhaustive' in step:
            search_task = Task.create(
                'run_exhaustive_search',
                run_exhaustive_search,
                step,
                as_fullblock,
                mfl,
                rank_options,
            )
        elif 'stepwise' in step:
            search_task = Task.create(
                'run_stepwise_search',
                run_stepwise_search,
                step,
                as_fullblock,
                mfl,
                rank_options,
            )
        else:
            raise NotImplementedError
        predecessors = [init_search_task] if not search_tasks else None
        wb.add_task(search_task, predecessors=predecessors)
        search_tasks.append(search_task)

    end_search_task = Task.create('end_search', end_search)
    wb.add_task(end_search_task)
    search_tasks.append(end_search_task)

    return wb, search_tasks


def insert_linearized_search_workflow(wb, model, steps_to_run, mfl, as_fullblock, rank_options):
    mfl_expanded = expand_model_features(model, mfl)

    start_task = wb.output_tasks[0]

    create_base_task = Task.create(
        'create_base_model', _create_base_model_entry, 'linearize', mfl_expanded, as_fullblock
    )
    wb.add_task(create_base_task, predecessors=[start_task])

    init_search_task = Task.create('init_search', init_search)
    wb.add_task(init_search_task, predecessors=[create_base_task])

    search_tasks = []
    for i, step in enumerate(steps_to_run, 1):
        if 'exhaustive' in step:
            search_task = Task.create(
                'run_exhaustive_search_linearized',
                run_exhaustive_search_linearized,
                step,
                i,
                as_fullblock,
                mfl_expanded,
                rank_options,
            )
        elif 'stepwise' in step:
            search_task = Task.create(
                'run_stepwise_search_linearized',
                run_stepwise_search_linearized,
                step,
                i,
                as_fullblock,
                mfl,
                rank_options,
            )
        else:
            raise NotImplementedError
        predecessors = [start_task, init_search_task] if not search_tasks else [start_task]
        wb.add_task(search_task, predecessors=predecessors)
        search_tasks.append(search_task)

    end_search_task = Task.create('end_search', end_search)
    wb.add_task(end_search_task)
    search_tasks.append(end_search_task)

    return wb, search_tasks


def run_exhaustive_search(
    context,
    step,
    as_fullblock,
    mfl,
    rank_options,
    base_model_entry_and_index_offset,
):
    base_model_entry, index_offset = base_model_entry_and_index_offset
    type = 'iiv' if 'no_of_etas' in step else 'covariance'
    mfl = expand_model_features(base_model_entry.model, mfl)

    _log_start_step(context, step)

    res = _run_exhaustive_search(
        context,
        type,
        mfl,
        rank_options,
        base_model_entry,
        index_offset,
        {'as_fullblock': as_fullblock},
    )
    best_model_entry, no_of_models, summary_tool = res

    _log_finish_step(context, step, best_model_entry)

    return (best_model_entry, no_of_models), summary_tool


def run_exhaustive_search_linearized(
    context,
    step,
    i,
    as_fullblock,
    mfl,
    rank_options,
    base_model_entry_and_index_offset,
    input_model_entry,
):
    base_model_entry, index_offset = base_model_entry_and_index_offset
    if 'no_of_etas' in step:
        type = 'iiv'
        mfl = mfl.iiv
    else:
        type = 'covariance'
        mfl = mfl.covariance
    mfl = expand_model_features(base_model_entry.model, mfl)

    linbase_model_entry = create_linearized_model_entry(
        context, as_fullblock, i, type, base_model_entry, input_model_entry
    )
    param_mapping = create_param_mapping(base_model_entry.model)

    _log_start_step(context, step)

    res = _run_exhaustive_search(
        context,
        type,
        mfl,
        rank_options,
        linbase_model_entry,
        index_offset,
        {'param_mapping': param_mapping},
    )
    best_model_entry, no_of_models, summary_tool = res

    delin_model_entry, delin_summary = create_delinearized_model_entry(
        context, rank_options, i, base_model_entry, param_mapping, best_model_entry
    )
    delin_summary = add_parent_column(delin_summary, [base_model_entry, delin_model_entry])
    tool_summaries = summary_tool + [delin_summary]
    _log_finish_step(context, step, delin_model_entry)

    return (delin_model_entry, no_of_models), tool_summaries


def _run_exhaustive_search(
    context,
    type,
    mfl,
    rank_options,
    base_model_entry,
    index_offset,
    kwargs=None,
):
    kwargs = kwargs if kwargs else dict()
    wf_step = algorithms.td_exhaustive(type, base_model_entry, mfl, index_offset, **kwargs)
    if not wf_step:
        context.log_info('No models to run, skipping step')
        return base_model_entry, index_offset, []
    mes = context.call_workflow(wf_step, 'run_candidates')
    rank_res = rank_models(
        context,
        rank_options,
        base_model_entry.model,
        [base_model_entry] + list(mes),
    )
    mes_all = (base_model_entry,) + mes
    best_model_entry = get_best_model_entry(mes_all, rank_res.final_model)
    summary_tool = add_parent_column(rank_res.summary_tool, mes_all)
    return best_model_entry, len(mes), [summary_tool]


def run_stepwise_search(
    context, step, as_fullblock, mfl, rank_options, base_model_entry_and_index_offset
):
    base_model_entry, index_offset = base_model_entry_and_index_offset

    _log_start_step(context, step)

    res = _run_stepwise_search(
        context, step, mfl, rank_options, base_model_entry, index_offset, as_fullblock=as_fullblock
    )
    best_model_entry, no_of_models, tool_summaries = res

    _log_finish_step(context, step, best_model_entry)

    return (best_model_entry, no_of_models), tool_summaries


def run_stepwise_search_linearized(
    context,
    step,
    i,
    as_fullblock,
    mfl,
    rank_options,
    base_model_entry_and_index_offset,
    input_model_entry,
):
    base_model_entry, index_offset = base_model_entry_and_index_offset
    type = 'iiv' if 'no_of_etas' in step else 'covariance'
    linbase_model_entry = create_linearized_model_entry(
        context, as_fullblock, i, type, base_model_entry, input_model_entry
    )
    param_mapping = create_param_mapping(base_model_entry.model)

    _log_start_step(context, step)

    res = _run_stepwise_search(
        context,
        f'{step}_linearized',
        mfl,
        rank_options,
        linbase_model_entry,
        index_offset,
        param_mapping=param_mapping,
    )
    best_model_entry, no_of_models, tool_summaries = res

    delin_model_entry, delin_summary = create_delinearized_model_entry(
        context, rank_options, i, base_model_entry, param_mapping, best_model_entry
    )
    delin_summary = add_parent_column(delin_summary, [base_model_entry, delin_model_entry])
    tool_summaries.append(delin_summary)

    _log_finish_step(context, step, best_model_entry)

    return (delin_model_entry, no_of_models), tool_summaries


def _run_stepwise_search(
    context,
    algorithm_name,
    mfl,
    rank_options,
    base_model_entry,
    index_offset,
    as_fullblock=False,
    param_mapping=None,
):
    algorithm_func = getattr(algorithms, algorithm_name)
    if not param_mapping:
        rank_res, mes = algorithm_func(
            context, base_model_entry, mfl, index_offset, as_fullblock, rank_options
        )
    else:
        rank_res, mes = algorithm_func(
            context, base_model_entry, mfl, index_offset, rank_options, param_mapping
        )
    if not rank_res:
        context.log_info('No models to run, skipping step')
        return base_model_entry, index_offset, []
    mes_all = (base_model_entry,) + mes
    tool_summaries = [add_parent_column(res.summary_tool, mes_all) for res in rank_res]
    best_model_entry = get_best_model_entry(mes_all, rank_res[-1].final_model)
    return best_model_entry, len(mes), tool_summaries


def compare_to_input_model(
    context,
    rank_options,
    best_model_entry,
    input_model_entry,
):
    input_model, input_res = input_model_entry.model, input_model_entry.modelfit_results
    best_model, best_res = best_model_entry.model, best_model_entry.modelfit_results

    if input_model != best_model and input_res and best_res:
        context.log_info('Comparing final model to input model')
        rank_res = rank_models(
            context,
            rank_options,
            input_model_entry.model,
            [input_model_entry, best_model_entry],
        )
        if rank_res.final_model == input_model:
            context.log_warning(
                f'Worse {rank_options.rank_type} in final model {best_model.name} '
                f'than {input_model.name}, selecting input model'
            )
        summary_tool = add_parent_column(
            rank_res.summary_tool, [input_model_entry, best_model_entry]
        )
    else:
        summary_tool = None

    return best_model_entry, summary_tool


def create_linearized_model_entry(
    context, as_fullblock, i, type, base_model_entry, input_model_entry
):
    lin_model_entry = run_linearization(
        context, base_model_entry, type, input_model_entry.modelfit_results
    )
    context.store_model_entry(lin_model_entry)
    lin_model = update_linearized_base_model(as_fullblock, input_model_entry, lin_model_entry)
    lin_model = lin_model.replace(name=f'linbase{i}')
    lin_model_entry = ModelEntry.create(
        lin_model, modelfit_results=None, parent=base_model_entry.model
    )

    wb = WorkflowBuilder(name='run_lin_base_model')
    wf_fit = create_fit_workflow(lin_model_entry)
    wb.insert_workflow(wf_fit)
    lin_model_entry = context.call_workflow(Workflow(wb), unique_name='run_base')

    return lin_model_entry


def run_linearization(context, baseme, type, results=None):
    if not results:
        results = baseme.modelfit_results
    description = create_description(baseme.model, type)
    linear_results = run_subtool(
        'linearize',
        context,
        model=baseme.model,
        results=results,
        description=description,
    )
    linbaseme = ModelEntry.create(
        model=linear_results.final_model, modelfit_results=linear_results.final_model_results
    )
    return linbaseme


def update_linearized_base_model(as_fullblock, inputme, baseme):
    added_params = baseme.model.parameters - inputme.model.parameters
    model = unfix_parameters(baseme.model, added_params.names)
    model = set_initial_estimates(model, baseme.modelfit_results.parameter_estimates)
    if as_fullblock:
        model = create_joint_distribution(
            model, individual_estimates=baseme.modelfit_results.individual_estimates
        )
    model = model.replace(name="linbase", description=baseme.model.description)
    return model


def create_delinearized_model_entry(
    context,
    rank_options,
    step_no,
    base_model_entry,
    param_mapping,
    best_model_entry,
):
    dl_wf = create_delinearize_workflow(
        base_model_entry.model, best_model_entry.model, param_mapping, step_no
    )
    context.log_info('Running delinearized model')
    dlin_model_entry = context.call_workflow(Workflow(dl_wf), "running_delinearization")

    rank_res = rank_models(
        context,
        rank_options,
        dlin_model_entry.model,
        [dlin_model_entry],
    )

    if not rank_res.final_model:
        context.abort_workflow('Delinearized model failed strictness criteria')

    summary_tool = add_parent_column(rank_res.summary_tool, [base_model_entry, dlin_model_entry])

    return dlin_model_entry, summary_tool


def create_delinearize_workflow(input_model, final_model, param_mapping, stepno=None):
    flm_etas = final_model.random_variables.iiv.names
    final_param_map = {k: v for k, v in param_mapping.items() if k in flm_etas}
    final_delinearized_model = delinearize_model(final_model, input_model, final_param_map)
    name = f'delinearized{stepno}'

    final_delinearized_model = final_delinearized_model.replace(
        name=name,
        description=final_model.description,
    )

    lin_model_entry = ModelEntry.create(model=final_delinearized_model, parent=input_model)
    dl_wf = WorkflowBuilder(name="delinearization_workflow")
    fit_wf = create_fit_workflow(lin_model_entry)
    dl_wf.insert_workflow(fit_wf)

    return dl_wf


def needs_base_model(model, mfl, as_fullblock, base_type):
    if base_type == 'bu':
        return True
    if (
        as_fullblock
        and len(model.random_variables.iiv.names) > 1
        and len(model.random_variables.iiv) != 1
    ):
        return True
    in_search_space = is_in_search_space(model, mfl, type='iiv')
    if mfl.covariance:
        return not (in_search_space and is_in_search_space(model, mfl, type='covariance'))
    return not in_search_space


def _create_base_model_entry(context, type, mfl, as_fullblock, input_model_entry):
    context.log_info("Creating base model")
    base_model_entry = create_base_model_entry(type, mfl, as_fullblock, input_model_entry)
    return base_model_entry


def create_base_model_entry(type, mfl, as_fullblock, input_model_entry):
    if type == 'td':
        base_model = _create_base_model_top_down(mfl, as_fullblock, input_model_entry)
    elif type == 'bu':
        base_model = _create_base_model_bottom_up(mfl, as_fullblock, input_model_entry)
    elif type == 'linearize':
        base_model = _create_base_model_linearize(mfl, input_model_entry)
    else:
        raise ValueError(f'Unknown base model type: {type}')
    return ModelEntry.create(base_model, parent=input_model_entry.model)


def _create_base_model_top_down(mfl, as_fullblock, input_model_entry):
    base_model = _create_base_model(
        input_model_entry, mfl.iiv.force_optional() + mfl.covariance, as_fullblock
    )
    return base_model


def _create_base_model_bottom_up(mfl, as_fullblock, input_model_entry):
    iivs = mfl.iiv - mfl.iiv.filter(filter_on='optional')
    iiv_params = [iiv.parameter for iiv in iivs]
    covs = ModelFeatures.create(
        [c for c in mfl.covariance if all(p in iiv_params for p in c.parameters)]
    )
    base_model = _create_base_model(input_model_entry, iivs + covs, as_fullblock)
    return base_model


def _create_base_model_linearize(mfl, input_model_entry):
    base_model = _create_base_model(
        input_model_entry, mfl.iiv.force_optional() + mfl.covariance, False, linearize=True
    )
    input_model = input_model_entry.model
    new_parameters = base_model.parameters - input_model.parameters
    if new_parameters:
        base_model = set_initial_estimates(base_model, {p: 0.000001 for p in new_parameters.names})
        base_model = fix_parameters(base_model, new_parameters.names)
    return base_model


def _create_base_model(input_model_entry, mfl, as_fullblock, linearize=False):
    input_model, input_res = input_model_entry.model, input_model_entry.modelfit_results
    base_model = update_initial_estimates(
        input_model,
        input_res,
        move_est_close_to_bounds=not linearize,
    )
    base_model = transform_into_search_space(base_model, mfl, type='iiv')
    base_model = transform_into_search_space(base_model, mfl, type='covariance')
    if as_fullblock and len(base_model.random_variables.iiv) > 1:
        ies = input_res.individual_estimates
        base_model = create_joint_distribution(base_model, individual_estimates=ies)
    base_mfl = get_model_features(base_model)
    description = create_description(base_mfl, type='iiv')
    base_model = base_model.replace(name='base', description=description)
    return base_model


def create_param_mapping(model):
    dists = model.random_variables.iiv
    param_subs = {
        parameter.symbol: parameter.init for parameter in model.parameters if parameter.fix
    }
    param_mapping = {}
    symbs_before_ode = [symb.name for symb in model.statements.before_odes.free_symbols]
    for eta in dists.names:
        if dists[eta].get_variance(eta).subs(param_subs) != 0:
            if eta not in symbs_before_ode:
                continue
            param_mapping[eta] = get_rv_parameters(model, eta)[0]
    return param_mapping


def init_search(base_model_entry):
    return base_model_entry, 0


def end_search(best_model_entry_and_index_offset):
    best_model_entry, _ = best_model_entry_and_index_offset
    return best_model_entry


def unpack_tool_summaries(*tool_summaries):
    return [tool_summary for step in tool_summaries for tool_summary in step]


def _log_start_step(context, step):
    context.log_info(f"Starting step '{step}'")


def _log_finish_step(context, step, best_model_entry):
    best_description = best_model_entry.model.description
    best_ofv = best_model_entry.modelfit_results.ofv
    context.log_info(f"Finished step '{step}'. Best model: {best_description}, OFV: {best_ofv:.3f}")


def postprocess(
    context,
    tool_summaries,
    model_entry_and_final_comparison,
):
    best_model_entry, final_summary_tool = model_entry_and_final_comparison
    tool_summaries += [final_summary_tool]

    summary_models = summarize_modelfit_results(context)

    context.store_final_model_entry(best_model_entry)

    best_model, best_res = best_model_entry.model, best_model_entry.modelfit_results
    plots = create_plots(best_model, best_res)
    eta_shrinkage_table = table_final_eta_shrinkage(best_model, best_res)

    res = IIVSearchResults(
        summary_tool=combine_summaries(tool_summaries, idx_start=1),
        summary_models=summary_models,
        final_model=best_model,
        final_results=best_res,
        final_model_dv_vs_ipred_plot=plots['dv_vs_ipred'],
        final_model_dv_vs_pred_plot=plots['dv_vs_pred'],
        final_model_cwres_vs_idv_plot=plots['cwres_vs_idv'],
        final_model_abs_cwres_vs_ipred_plot=plots['abs_cwres_vs_ipred'],
        final_model_eta_distribution_plot=plots['eta_distribution'],
        final_model_eta_shrinkage=eta_shrinkage_table,
    )

    return res


def combine_summaries(summaries, idx_start):
    keys = list(range(idx_start, idx_start + len(summaries)))
    return concat_summaries(summaries, keys=keys)


def _results(context, res):
    context.log_info("Finishing tool iivsearch")
    return res


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    model,
    results,
    algorithm,
    search_space,
    as_fullblock,
    rank_type,
    linearize,
    cutoff,
    strictness,
    correlation_algorithm,
    E_p,
    E_q,
    parameter_uncertainty_method,
):
    try:
        ModelFeatures.create(search_space)
    except ValueError as e:
        raise ValueError(f'Could not parse `search_space`: {search_space}\n\t{e}')

    try:
        get_model_features(model, type='iiv')
    except NotImplementedError:
        raise ValueError('Invalid `model`: could not determine eta distributions')

    if (
        strictness is not None
        and parameter_uncertainty_method is None
        and "rse" in strictness.lower()
    ):
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
