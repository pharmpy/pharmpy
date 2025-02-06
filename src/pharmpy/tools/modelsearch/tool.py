from dataclasses import dataclass
from typing import Literal, Optional, Union

import pharmpy.tools.modelsearch.algorithms as algorithms
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.tools.common import RANK_TYPES, ToolResults, create_results, update_initial_estimates
from pharmpy.tools.mfl.least_number_of_transformations import least_number_of_transformations
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import calculate_mbic_penalty, summarize_modelfit_results_from_entries
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults

from ..mfl.parse import parse as mfl_parse
from .algorithms import _add_allometry
from .filter import mfl_filtering


def create_workflow(
    model: Model,
    results: ModelfitResults,
    search_space: Union[str, ModelFeatures],
    algorithm: Literal[tuple(algorithms.ALGORITHMS)] = 'reduced_stepwise',
    iiv_strategy: Literal[tuple(algorithms.IIV_STRATEGIES)] = 'absorption_delay',
    rank_type: Literal[tuple(RANK_TYPES)] = 'bic',
    cutoff: Optional[Union[float, int]] = None,
    strictness: str = "minimization_successful or (rounding_errors and sigdigs >= 0.1)",
    E: Optional[Union[float, str]] = None,
):
    """Run Modelsearch tool. For more details, see :ref:`modelsearch`.

    Parameters
    ----------
    model : Model
        Pharmpy model
    results : ModelfitResults
        Results for model
    search_space : str, ModelFeatures
        Search space to test. Either as a string or a ModelFeatures object.
    algorithm : {'exhaustive', 'exhaustive_stepwise', 'reduced_stepwise'}
        Algorithm to use.
    iiv_strategy : {'no_add', 'add_diagonal', 'fullblock', 'absorption_delay'}
        If/how IIV should be added to candidate models. Default is 'absorption_delay'.
    rank_type : {'ofv', 'lrt', 'aic', 'bic', 'mbic'}
        Which ranking type should be used. Default is BIC.
    cutoff : float
        Cutoff for which value of the ranking function that is considered significant. Default
        is None (all models will be ranked)
    strictness : str or None
        Strictness criteria
    E : float
        Expected number of predictors (used for mBIC). Must be set when using mBIC

    Returns
    -------
    ModelSearchResults
        Modelsearch tool result object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import run_modelsearch, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> res = load_example_modelfit_results("pheno")
    >>> search_space = 'ABSORPTION(ZO);PERIPHERALS(1)'
    >>> run_modelsearch(model=model, results=res, search_space=search_space, algorithm='exhaustive') # doctest: +SKIP

    """
    wb = WorkflowBuilder(name='modelsearch')
    start_task = Task(
        'start_modelsearch',
        start,
        search_space,
        algorithm,
        iiv_strategy,
        rank_type,
        cutoff,
        results,
        model,
        strictness,
        E,
    )
    wb.add_task(start_task)
    task_results = Task('results', _results)
    wb.add_task(task_results, predecessors=[start_task])
    return Workflow(wb)


def start(
    context,
    search_space,
    algorithm,
    iiv_strategy,
    rank_type,
    cutoff,
    results,
    model,
    strictness,
    E,
):
    context.log_info("Starting tool modelsearch")

    model = model.replace(name="input", description="")
    context.store_input_model_entry(ModelEntry.create(model=model, modelfit_results=results))
    context.log_info(f"Input model OFV: {results.ofv:.3f}")

    wb = WorkflowBuilder()

    start_task = Task('start_modelsearch', _start, model, results)
    wb.add_task(start_task)

    algorithm_func = getattr(algorithms, algorithm)

    if isinstance(search_space, str):
        mfl_statements = mfl_parse(search_space, mfl_class=True)
    else:
        mfl_statements = search_space

    if mfl_statements.allometry is not None:
        mfl_allometry = mfl_statements.allometry
        mfl_statements = mfl_statements.replace(allometry=None)
    else:
        mfl_allometry = None

    # Add base model task
    model_mfl = get_model_features(model, supress_warnings=True)
    model_mfl = ModelFeatures.create_from_mfl_string(model_mfl)
    if not mfl_statements.contain_subset(model_mfl, tool="modelsearch") or mfl_allometry:
        context.log_info("Creating base model")
        base_task = Task("create_base_model", create_base_model, mfl_statements, mfl_allometry)
        wb.add_task(base_task, predecessors=start_task)

        base_fit = create_fit_workflow(n=1)
        wb.insert_workflow(base_fit, predecessors=base_task)
        base_fit = base_fit.output_tasks
    else:
        # Always create base, even if same as input for proper
        # post-process behaviour.
        base_fit = start_task

    task_result = Task(
        'results',
        post_process,
        search_space,
        rank_type,
        cutoff,
        strictness,
        E,
        context,
    )

    # Filter the mfl_statements from base model attributes
    mfl_funcs = filter_mfl_statements(
        mfl_statements, create_base_model(mfl_statements, mfl_allometry, model)
    )

    # TODO : Implement task for filtering the search space instead
    wf_search, candidate_model_tasks = algorithm_func(
        mfl_funcs, iiv_strategy, allometry=mfl_allometry
    )

    if candidate_model_tasks:
        # Clear base description to not interfere with candidate models
        base_clear_task = Task("clear_base_description", clear_description)
        wb.add_task(base_clear_task, predecessors=base_fit)

        wb.insert_workflow(wf_search, predecessors=base_clear_task)

        if base_fit != start_task:
            wb.add_task(task_result, predecessors=[start_task] + base_fit + candidate_model_tasks)
        else:
            wb.add_task(task_result, predecessors=[start_task] + candidate_model_tasks)
    else:
        if base_fit != start_task:
            wb.add_task(task_result, predecessors=[start_task] + base_fit + candidate_model_tasks)
        else:
            wb.add_task(task_result, predecessors=[start_task] + candidate_model_tasks)

    context.log_info(f"Starting algorithm '{algorithm}'")
    res = context.call_workflow(wb, 'run_candidate_models')
    context.log_info(
        f"Finished algorithm '{algorithm}'. Best model: "
        f"{res.final_model.name}, OFV: {res.final_results.ofv:.3f}"
    )

    if res.final_model.name == model.name:
        context.log_warning(
            f'Worse {rank_type} in final model {res.final_model.name} '
            f'than {model.name}, selecting input model'
        )
    context.store_final_model_entry(res.final_model)

    return res


def _start(model, modelfit_results):
    return ModelEntry.create(model, modelfit_results=modelfit_results)


def _results(context, res):
    context.log_info("Finishing tool modelsearch")
    return res


def clear_description(model_entry):
    model, res, parent = model_entry.model, model_entry.modelfit_results, model_entry.parent
    return ModelEntry.create(model.replace(description=""), modelfit_results=res, parent=parent)


def filter_mfl_statements(mfl_statements: ModelFeatures, model_entry: ModelEntry):
    model = model_entry.model
    ss_funcs = mfl_statements.convert_to_funcs()
    model_mfl = ModelFeatures.create_from_mfl_string(get_model_features(model))
    model_funcs = model_mfl.convert_to_funcs()
    res = {k: ss_funcs[k] for k in set(ss_funcs) - set(model_funcs)}
    return {k: v for k, v in sorted(res.items(), key=lambda x: (x[0][0], x[0][1]))}


def create_base_model(ss, allometry, model_or_model_entry):
    if isinstance(model_or_model_entry, ModelEntry):
        model = model_or_model_entry.model
        res = model_or_model_entry.modelfit_results
    else:
        model = model_or_model_entry
        res = None

    base = update_initial_estimates(model, res) if res else model

    model_mfl = get_model_features(model, supress_warnings=True)
    model_mfl = ModelFeatures.create_from_mfl_string(model_mfl)
    added_features = ""
    lnt = least_number_of_transformations(model_mfl, ss, tool="modelsearch")
    for name, func in lnt.items():
        base = func(base)
        added_features += f';{name[0]}({name[1]})'
    base = base.replace(name="base", description=added_features[1:])
    base = _add_allometry(base, allometry)

    return ModelEntry.create(base, modelfit_results=None, parent=model)


def post_process(mfl, rank_type, cutoff, strictness, E, context, *model_entries):
    input_model_entry, base_model_entry, res_model_entries = categorize_model_entries(model_entries)

    summary_models = summarize_modelfit_results_from_entries(model_entries)
    summary_models['step'] = [0] + [1] * (len(summary_models) - 1)
    summary_models = summary_models.reset_index().set_index(['step', 'model'])

    if rank_type == 'mbic':
        penalties = [
            calculate_mbic_penalty(me.model, mfl, E_p=E)
            for me in [base_model_entry] + res_model_entries
        ]
    else:
        penalties = None

    res = create_results(
        ModelSearchResults,
        input_model_entry,
        base_model_entry,
        res_model_entries,
        rank_type,
        cutoff,
        summary_models=summary_models,
        strictness=strictness,
        penalties=penalties,
        context=context,
    )
    return res


def categorize_model_entries(model_entries):
    res_model_entries = []
    input_model_entry = None
    base_model_entry = None
    for model_entry in model_entries:
        model = model_entry.model
        if model.name.startswith('modelsearch_run'):
            res_model_entries.append(model_entry)
        elif model.name == "base":
            base_model_entry = model_entry
        else:
            input_model_entry = model_entry
    if not input_model_entry:
        raise ValueError('Error in workflow: No input model')
    if not base_model_entry:
        input_model_entry = input_model_entry
        base_model_entry = input_model_entry

    return input_model_entry, base_model_entry, res_model_entries


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    search_space,
    algorithm,
    iiv_strategy,
    rank_type,
    model,
    strictness,
    E,
):
    if isinstance(search_space, str):
        try:
            statements = mfl_parse(search_space)
        except:  # noqa E722
            raise ValueError(f'Invalid `search_space`, could not be parsed: "{search_space}"')
    else:
        statements = search_space.filter("pk").mfl_statement_list()

    modelsearch_statements = mfl_filtering(statements, 'modelsearch')
    bad_statements = list(
        filter(lambda statement: statement not in modelsearch_statements, statements)
    )

    if bad_statements:
        raise ValueError(
            f'Invalid `search_space`: found unknown statement of type {type(bad_statements[0]).__name__}.'
        )

    allometry = ModelFeatures.create_from_mfl_statement_list(statements).allometry
    if allometry:
        covariate = allometry.covariate
        if covariate not in list(model.dataset.columns):
            raise ValueError(
                f'Invalid `search_space`: allometric variable \'{covariate}\' not in dataset'
            )

    if strictness is not None and "rse" in strictness.lower():
        if model.execution_steps[-1].parameter_uncertainty_method is None:
            raise ValueError(
                '`parameter_uncertainty_method` not set for model, cannot calculate relative standard errors.'
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
class ModelSearchResults(ToolResults):
    pass
