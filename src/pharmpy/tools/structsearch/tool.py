from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling.tmdd import DV_TYPES
from pharmpy.tools.common import (
    RANK_TYPES,
    ToolResults,
    add_parent_column,
    concat_summaries,
    create_plots,
    flatten_list,
    table_final_eta_shrinkage,
    update_initial_estimates,
)
from pharmpy.tools.mfl.parse import ModelFeatures
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import (
    run_subtool,
    summarize_errors_from_entries,
    summarize_modelfit_results_from_entries,
)
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults

from .drugmetabolite import create_drug_metabolite_models
from .pkpd import create_baseline_pd_model, create_pkpd_models
from .tmdd import create_qss_models, create_remaining_models

TYPES = frozenset(('pkpd', 'drug_metabolite', 'tmdd'))


def create_workflow(
    model: Model,
    results: ModelfitResults,
    type: Literal[tuple(TYPES)],
    search_space: Optional[Union[str, ModelFeatures]] = None,
    b_init: Optional[Union[int, float]] = None,
    emax_init: Optional[Union[int, float]] = None,
    ec50_init: Optional[Union[int, float]] = None,
    met_init: Optional[Union[int, float]] = None,
    extra_model: Optional[Model] = None,
    rank_type: Literal[tuple(RANK_TYPES)] = 'bic',
    cutoff: Optional[Union[float, int]] = None,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs >= 0.1)",
    extra_model_results: Optional[ModelfitResults] = None,
    dv_types: Optional[dict[Literal[DV_TYPES], int]] = None,
    parameter_uncertainty_method: Optional[Literal['SANDWICH', 'SMAT', 'RMAT', 'EFIM']] = None,
):
    """Run the structsearch tool. For more details, see :ref:`structsearch`.

    Parameters
    ----------
    model : Model
        Pharmpy start model
    results : ModelfitResults
        Results for the start model
    type : str
        Type of model. Currently only 'drug_metabolite' and 'pkpd'
    search_space : str
        Search space to test
    b_init: float
        Initial estimate for the baseline for pkpd models.
    emax_init: float
        Initial estimate for E_MAX (for pkpd models only).
    ec50_init: float
        Initial estimate for EC_50 (for pkpd models only).
    met_init: float
        Initial estimate for MET (for pkpd models only).
    extra_model : Model
        Optional extra Pharmpy model to use in TMDD structsearch
    extra_model_results : ModelfitResults
        Results for the extra model
    rank_type : {'ofv', 'lrt', 'aic', 'bic'}
        Which ranking type should be used. Default is BIC.
    cutoff : float
        Cutoff for which value of the ranking function that is considered significant. Default
        is None (all models will be ranked)
    strictness : str or None
        Strictness criteria
    dv_types : dict
        Dictionary of DV types for TMDD models with multiple DVs
    parameter_uncertainty_method : {'SANDWICH', 'SMAT', 'RMAT', 'EFIM'} or None
        Parameter uncertainty method. Will be used in ranking models if strictness includes
        parameter uncertainty

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
    >>> run_structsearch(model=model, results=results, model_type='pkpd')   # doctest: +SKIP
    """

    wb = WorkflowBuilder(name="structsearch")
    if type == 'tmdd':
        start_task = Task(
            'run_tmdd',
            run_tmdd,
            model,
            results,
            extra_model,
            extra_model_results,
            rank_type,
            cutoff,
            strictness,
            parameter_uncertainty_method,
            dv_types,
        )
    elif type == 'pkpd':
        start_task = Task(
            'run_pkpd',
            run_pkpd,
            model,
            results,
            search_space,
            b_init,
            emax_init,
            ec50_init,
            met_init,
            rank_type,
            cutoff,
            strictness,
            parameter_uncertainty_method,
        )
    elif type == 'drug_metabolite':
        start_task = Task(
            'run_drug_metabolite',
            run_drug_metabolite,
            model,
            search_space,
            results,
            rank_type,
            cutoff,
            strictness,
            parameter_uncertainty_method,
        )
    wb.add_task(start_task)
    return Workflow(wb)


def run_tmdd(
    context,
    model,
    results,
    extra_model,
    extra_model_results,
    rank_type,
    cutoff,
    strictness,
    parameter_uncertainty_method,
    dv_types,
):
    context.log_info("Starting tool structsearch")
    model = store_input_model(context, model, results)

    model = update_initial_estimates(model, results)
    model_entry = ModelEntry.create(model, modelfit_results=results)

    qss_candidate_models = create_qss_models(model, results.parameter_estimates, dv_types)
    qss_candidate_entries = [
        ModelEntry.create(m, modelfit_results=None, parent=model) for m in qss_candidate_models
    ]

    if extra_model is not None:
        extra_model = update_initial_estimates(extra_model, extra_model_results)
        extra_qss_candidate_models = create_qss_models(
            extra_model,
            extra_model_results.parameter_estimates,
            dv_types,
            index=len(qss_candidate_models),
        )
        extra_qss_candidate_entries = [
            ModelEntry.create(model, modelfit_results=None, parent=extra_model)
            for model in extra_qss_candidate_models
        ]
        qss_candidate_entries += extra_qss_candidate_entries

    wf = create_fit_workflow(qss_candidate_entries)
    wb = WorkflowBuilder(wf)
    task_results = Task('results', bundle_results)
    wb.add_task(task_results, predecessors=wf.output_tasks)
    qss_run_entries = context.call_workflow(Workflow(wb), 'results_QSS')

    rank_res_step_1 = rank_models(
        context,
        model_entry,
        list(qss_run_entries),
        rank_type,
        cutoff,
        strictness,
        parameter_uncertainty_method,
    )

    best_qss_entry = ModelEntry.create(
        rank_res_step_1.final_model, modelfit_results=rank_res_step_1.final_results
    )

    models = create_remaining_models(
        model,
        best_qss_entry.modelfit_results.parameter_estimates,
        len(best_qss_entry.model.statements.ode_system.find_peripheral_compartments()),
        dv_types,
        len(qss_candidate_entries),
    )
    remaining_model_entries = [
        ModelEntry.create(model, modelfit_results=None, parent=best_qss_entry.model)
        for model in models
    ]

    wf2 = create_fit_workflow(remaining_model_entries)
    wb2 = WorkflowBuilder(wf2)
    task_results = Task('results', bundle_results)
    wb2.add_task(task_results, predecessors=wf2.output_tasks)
    run_model_entries = context.call_workflow(Workflow(wb2), 'results_remaining')

    rank_res_step_2 = rank_models(
        context,
        best_qss_entry,
        list(run_model_entries),
        rank_type,
        cutoff,
        strictness,
        parameter_uncertainty_method,
    )

    model_entries = [model_entry] + list(qss_run_entries) + list(run_model_entries)
    summary_step_1 = add_parent_column(rank_res_step_1.summary_tool, model_entries)
    summary_step_2 = add_parent_column(rank_res_step_2.summary_tool, model_entries)
    summary_tool = concat_summaries([summary_step_1, summary_step_2], keys=[1, 2])

    tables = create_result_tables([[model_entry], list(qss_run_entries), list(run_model_entries)])
    plots = create_plots(rank_res_step_2.final_model, rank_res_step_2.final_results)
    eta_shrinkage = table_final_eta_shrinkage(
        rank_res_step_2.final_model, rank_res_step_2.final_results
    )

    res = StructSearchResults(
        summary_tool=summary_tool,
        summary_models=tables['summary_models'],
        summary_errors=tables['summary_errors'],
        final_model=rank_res_step_2.final_model,
        final_results=rank_res_step_2.final_results,
        final_model_dv_vs_ipred_plot=plots['dv_vs_ipred'],
        final_model_dv_vs_pred_plot=plots['dv_vs_pred'],
        final_model_cwres_vs_idv_plot=plots['cwres_vs_idv'],
        final_model_abs_cwres_vs_ipred_plot=plots['abs_cwres_vs_ipred'],
        final_model_eta_distribution_plot=plots['eta_distribution'],
        final_model_eta_shrinkage=eta_shrinkage,
    )

    final_model = res.final_model.replace(name="final")
    context.store_final_model_entry(final_model)

    context.log_info("Finishing tool structsearch")
    return res


def rank_models(
    context,
    base_model_entry,
    candidate_model_entries,
    rank_type,
    cutoff,
    strictness,
    parameter_uncertainty_method,
):
    model_entries = [base_model_entry] + candidate_model_entries
    models = [me.model for me in model_entries]
    results = [me.modelfit_results for me in model_entries]

    rank_type = rank_type + '_mixed' if rank_type == 'bic' else rank_type

    rank_res = run_subtool(
        tool_name='modelrank',
        ctx=context,
        models=models,
        results=results,
        ref_model=base_model_entry.model,
        rank_type=rank_type,
        alpha=cutoff,
        strictness=strictness,
        parameter_uncertainty_method=parameter_uncertainty_method,
    )

    return rank_res


def create_result_tables(model_entries_per_step: list[list[ModelEntry]]):
    sum_models = [summarize_modelfit_results_from_entries(mes) for mes in model_entries_per_step]
    keys = range(0, len(model_entries_per_step))
    summary_models = concat_summaries(sum_models, keys)
    model_entries = flatten_list(model_entries_per_step)
    summary_errors = summarize_errors_from_entries(model_entries)

    tables = {
        'summary_models': summary_models,
        'summary_errors': summary_errors,
    }
    return tables


def run_pkpd(
    context,
    input_model,
    results,
    search_space,
    b_init,
    emax_init,
    ec50_init,
    met_init,
    rank_type,
    cutoff,
    strictness,
    parameter_uncertainty_method,
):
    context.log_info("Starting tool structsearch")
    input_model = store_input_model(context, input_model, results)

    model_entry = ModelEntry.create(input_model, modelfit_results=results)
    baseline_pd_model = create_baseline_pd_model(input_model, results.parameter_estimates, b_init)
    baseline_pd_model_entry = ModelEntry.create(baseline_pd_model, modelfit_results=None)

    wf = create_fit_workflow(baseline_pd_model_entry)
    wb = WorkflowBuilder(wf)
    task_results = Task('results2', bundle_results)
    wb.add_task(task_results, predecessors=wf.output_tasks)
    pd_baseline_fit = context.call_workflow(Workflow(wb), 'results_remaining')

    pkpd_models = create_pkpd_models(
        input_model,
        search_space,
        b_init,
        results.parameter_estimates,
        emax_init,
        ec50_init,
        met_init,
    )
    pkpd_model_entries = [
        ModelEntry.create(model, modelfit_results=None, parent=baseline_pd_model)
        for model in pkpd_models
    ]

    wf2 = create_fit_workflow(pkpd_model_entries)
    wb2 = WorkflowBuilder(wf2)
    task_results = Task('results2', bundle_results)
    wb2.add_task(task_results, predecessors=wf2.output_tasks)
    pkpd_models_fit = context.call_workflow(Workflow(wb2), 'results_remaining')

    rank_res = rank_models(
        context,
        pd_baseline_fit[0],
        list(pkpd_models_fit),
        rank_type,
        cutoff,
        strictness,
        parameter_uncertainty_method,
    )

    summary_tool = add_parent_column(
        rank_res.summary_tool, [pd_baseline_fit[0]] + list(pkpd_models_fit)
    )
    tables = create_result_tables([[model_entry], list(pd_baseline_fit), list(pkpd_models_fit)])
    plots = create_plots(rank_res.final_model, rank_res.final_results)

    eta_shrinkage = table_final_eta_shrinkage(rank_res.final_model, rank_res.final_results)

    res = StructSearchResults(
        summary_tool=summary_tool,
        summary_models=tables['summary_models'],
        summary_errors=tables['summary_errors'],
        final_model=rank_res.final_model,
        final_results=rank_res.final_results,
        final_model_dv_vs_ipred_plot=plots['dv_vs_ipred'],
        final_model_dv_vs_pred_plot=plots['dv_vs_pred'],
        final_model_cwres_vs_idv_plot=plots['cwres_vs_idv'],
        final_model_abs_cwres_vs_ipred_plot=plots['abs_cwres_vs_ipred'],
        final_model_eta_distribution_plot=plots['eta_distribution'],
        final_model_eta_shrinkage=eta_shrinkage,
    )

    final_model = res.final_model.replace(name="final")
    context.store_final_model_entry(final_model)

    context.log_info("Finishing tool structsearch")
    return res


def run_drug_metabolite(
    context,
    model,
    search_space,
    results,
    rank_type,
    cutoff,
    strictness,
    parameter_uncertainty_method,
):
    context.log_info("Starting tool structsearch")
    # Create links to input model
    model = store_input_model(context, model, results)

    model = update_initial_estimates(model, results)
    wb, candidate_model_tasks, base_model_description = create_drug_metabolite_models(
        model, results, search_space
    )

    task_results = Task(
        'Results',
        post_process_drug_metabolite,
        ModelEntry.create(model=model, modelfit_results=results),
        base_model_description,
        rank_type,
        cutoff,
        strictness,
        parameter_uncertainty_method,
    )

    wb.add_task(task_results, predecessors=candidate_model_tasks)
    results = context.call_workflow(Workflow(wb), "results_remaining")

    final_model = results.final_model.replace(name="final")
    context.store_final_model_entry(final_model)

    context.log_info("Finishing tool structsearch")
    return results


def post_process_drug_metabolite(
    context,
    input_model_entry,
    base_model_description,
    rank_type,
    cutoff,
    strictness,
    parameter_uncertainty_method,
    *model_entries,
):
    # NOTE : The base model is part of the model_entries but not the user_input_model
    base_model_entry, res_model_entries = categorize_drug_metabolite_model_entries(
        input_model_entry, model_entries, base_model_description
    )

    results_to_summarize = [[input_model_entry]]

    if input_model_entry != base_model_entry:
        results_to_summarize.append([base_model_entry])
    if res_model_entries:
        results_to_summarize.append(res_model_entries)

    rank_res = rank_models(
        context,
        base_model_entry,
        res_model_entries,
        rank_type,
        cutoff,
        strictness,
        parameter_uncertainty_method,
    )

    summary_tool = add_parent_column(rank_res.summary_tool, model_entries)
    tables = create_result_tables(results_to_summarize)
    plots = create_plots(rank_res.final_model, rank_res.final_results)
    eta_shrinkage = table_final_eta_shrinkage(rank_res.final_model, rank_res.final_results)

    res = StructSearchResults(
        summary_tool=summary_tool,
        summary_models=tables['summary_models'],
        summary_errors=tables['summary_errors'],
        final_model=rank_res.final_model,
        final_results=rank_res.final_results,
        final_model_dv_vs_ipred_plot=plots['dv_vs_ipred'],
        final_model_dv_vs_pred_plot=plots['dv_vs_pred'],
        final_model_cwres_vs_idv_plot=plots['cwres_vs_idv'],
        final_model_abs_cwres_vs_ipred_plot=plots['abs_cwres_vs_ipred'],
        final_model_eta_distribution_plot=plots['eta_distribution'],
        final_model_eta_shrinkage=eta_shrinkage,
    )

    return res


def categorize_drug_metabolite_model_entries(
    input_model_entry, model_entries, base_model_description
):
    # NOTE : The base model is part of the model_entries but not the user_input_model
    res_models = []
    base_model_entry = None
    for model_entry in model_entries:
        model = model_entry.model
        if model.description == base_model_description:
            model_entry = ModelEntry.create(
                model=model_entry.model, parent=None, modelfit_results=model_entry.modelfit_results
            )
            base_model_entry = model_entry
        else:
            res_models.append(model_entry)
    if not base_model_entry:
        # No base model found indicate user_input_model is base
        base_model_entry = input_model_entry
    if not input_model_entry:
        raise ValueError('Error in workflow: No input model')

    if base_model_entry != input_model_entry:
        # Change parent model to base model instead of temporary model names or input model
        res_models = [
            (
                me
                if me.parent.name
                not in ("TEMP", input_model_entry.model.name)  # TEMP name for drug-met models
                else ModelEntry.create(
                    model=me.model,
                    parent=base_model_entry.model,
                    modelfit_results=me.modelfit_results,
                )
            )
            for me in res_models
        ]

    return base_model_entry, res_models


def bundle_results(*args):
    return args


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    type,
    rank_type,
    cutoff,
    strictness,
    model,
    dv_types,
    search_space,
    b_init,
    emax_init,
    ec50_init,
    met_init,
    extra_model,
    extra_model_results,
    parameter_uncertainty_method,
):
    if (
        strictness is not None
        and parameter_uncertainty_method is None
        and "rse" in strictness.lower()
    ):
        if model.execution_steps[-1].parameter_uncertainty_method is None:
            raise ValueError(
                'parameter_uncertainty_method not set for model, cannot calculate relative standard errors.'
            )

    if type.lower() == 'tmdd':
        if search_space is not None:
            raise ValueError('Invalid argument "search_space" for TMDD models.')
        if any([b_init, emax_init, ec50_init, met_init]):
            raise ValueError(
                'Invalid arguments "b_init", "emax_init", "ec50_init" and "met_init" for TMDD models.'
            )
        if dv_types is not None:
            if not len(dv_types.values()) == len(set(dv_types.values())):
                raise ValueError('Values must be unique.')
            if len(dv_types) == 1:
                raise ValueError('`dv_types` must contain more than 1 dv type')
            for key, value in dv_types.items():
                if key not in ['drug', 'drug_tot'] and value == 1:
                    raise ValueError('Only drug can have DVID = 1. Please choose another DVID.')

    elif type.lower() == 'pkpd':
        if search_space is not None:
            if isinstance(search_space, str):
                search_space = mfl_parse(search_space, True)
            if search_space.filter("pd") != search_space:
                raise ValueError(
                    f'Argument search_space contain attributes not used for "{type}" models'
                )
        else:
            raise ValueError("Argument search_space need to be specified.")
        if extra_model is not None:
            raise ValueError('Invalid argument "extra_model" for PKPD models.')
        if extra_model_results is not None:
            raise ValueError('Invalid argument "extra_model_results" for PKPD models.')
        if dv_types is not None:
            raise ValueError('Invalid argument "dv_types" for PKPD models.')
        if b_init is None:
            raise ValueError("Initial estimate for baseline is needed")
        if emax_init is None:
            raise ValueError("Initial estimate for E_max is needed")
        if ec50_init is None:
            raise ValueError("Initial estimate for EC_50 is needed")
        if met_init is None:
            raise ValueError("Initial estimate for MET is needed")

    elif type.lower() == 'drug_metabolite':
        if search_space is not None:
            if isinstance(search_space, str):
                search_space = mfl_parse(search_space, True)
            if search_space.filter("metabolite") != search_space:
                raise ValueError(
                    f'Argument search_space contain attributes not used for "{type}" models'
                )
        else:
            raise ValueError("Argument search_space need to be specified.")
        if any([b_init, emax_init, ec50_init, met_init]):
            raise ValueError(
                'Invalid arguments "b_init", "emax_init", "ec50_init" and "met_init" for drug metabolite models.'
            )
        if extra_model is not None:
            raise ValueError('Invalid argument "extra_model" for drug metabolite models.')
        if extra_model_results is not None:
            raise ValueError('Invalid argument "extra_model_results" for drug metabolite models.')
        if dv_types is not None:
            raise ValueError('Invalid argument "dv_types" for drug metabolite models.')


def store_input_model(context, model, results):
    model = model.replace(name="input", description="")
    me = ModelEntry.create(model=model, modelfit_results=results)
    context.store_input_model_entry(me)
    return model


@dataclass(frozen=True)
class StructSearchResults(ToolResults):
    pass
