from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import pharmpy.tools.modelsearch.algorithms as algorithms
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import set_initial_estimates
from pharmpy.tools import get_model_features
from pharmpy.tools.common import RANK_TYPES, ToolResults, create_results
from pharmpy.tools.mfl.parse import ModelFeatures
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import summarize_modelfit_results_from_entries
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder, call_workflow
from pharmpy.workflows.results import ModelfitResults

from ..mfl.filter import mfl_filtering
from ..mfl.parse import parse as mfl_parse


def create_workflow(
    search_space: Union[str, ModelFeatures],
    algorithm: Literal[tuple(algorithms.ALGORITHMS)],
    iiv_strategy: Literal[tuple(algorithms.IIV_STRATEGIES)] = 'absorption_delay',
    rank_type: Literal[tuple(RANK_TYPES)] = 'mbic',
    cutoff: Optional[Union[float, int]] = None,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs >= 0.1)",
):
    """Run Modelsearch tool. For more details, see :ref:`modelsearch`.

    Parameters
    ----------
    search_space : str, ModelFeatures
        Search space to test. Either as a string or a ModelFeatures object.
    algorithm : {'exhaustive', 'exhaustive_stepwise', 'reduced_stepwise'}
        Algorithm to use.
    iiv_strategy : {'no_add', 'add_diagonal', 'fullblock', 'absorption_delay'}
        If/how IIV should be added to candidate models. Default is 'absorption_delay'.
    rank_type : {'ofv', 'lrt', 'aic', 'bic', 'mbic'}
        Which ranking type should be used. Default is mBIC.
    cutoff : float
        Cutoff for which value of the ranking function that is considered significant. Default
        is None (all models will be ranked)
    results : ModelfitResults
        Results for model
    model : Model
        Pharmpy model
    strictness : str or None
        Strictness criteria

    Returns
    -------
    ModelSearchResults
        Modelsearch tool result object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import run_modelsearch, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> run_modelsearch('ABSORPTION(ZO);PERIPHERALS(1)', 'exhaustive', results=results, model=model) # doctest: +SKIP

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
):
    wb = WorkflowBuilder()

    start_task = Task('start_modelsearch', _start, model, results)
    wb.add_task(start_task)

    algorithm_func = getattr(algorithms, algorithm)

    if isinstance(search_space, str):
        mfl_statements = mfl_parse(search_space, mfl_class=True)
    else:
        mfl_statements = search_space

    # Add base model task
    model_mfl = get_model_features(model, supress_warnings=True)
    model_mfl = ModelFeatures.create_from_mfl_string(model_mfl)
    if not mfl_statements.contain_subset(model_mfl, tool="modelsearch"):
        base_task = Task("create_base_model", create_base_model, mfl_statements)
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
        rank_type,
        cutoff,
        strictness,
        context,
    )

    # Filter the mfl_statements from base model attributes
    mfl_funcs = filter_mfl_statements(mfl_statements, create_base_model(mfl_statements, model))

    # TODO : Implement task for filtering the search space instead
    wf_search, candidate_model_tasks = algorithm_func(mfl_funcs, iiv_strategy)

    if candidate_model_tasks:
        # Clear base description to not interfere with candidate models
        base_clear_task = Task("Clear_base_description", clear_description)
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

    res = call_workflow(wb, 'run_candidate_models', context)

    return res


def _start(model, modelfit_results):
    return ModelEntry.create(model, modelfit_results=modelfit_results)


def _results(res):
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


def _update_results(base):
    """
    Changes the name and description of the connected ModelfitResults object
    to match the one found in the model object"""
    base_results = base.modelfit_results
    # Workaround due to not having a replace method
    return base.replace(
        modelfit_results=ModelfitResults(
            name=base.name,
            description=base.description,
            ofv=base_results.ofv,
            ofv_iterations=base_results.ofv_iterations,
            parameter_estimates=base_results.parameter_estimates,
            parameter_estimates_sdcorr=base_results.parameter_estimates_sdcorr,
            parameter_estimates_iterations=base_results.parameter_estimates_iterations,
            covariance_matrix=base_results.covariance_matrix,
            correlation_matrix=base_results.correlation_matrix,
            precision_matrix=base_results.precision_matrix,
            standard_errors=base_results.standard_errors,
            standard_errors_sdcorr=base_results.standard_errors_sdcorr,
            relative_standard_errors=base_results.relative_standard_errors,
            minimization_successful=base_results.minimization_successful,
            minimization_successful_iterations=base_results.minimization_successful_iterations,
            estimation_runtime=base_results.estimation_runtime,
            estimation_runtime_iterations=base_results.estimation_runtime_iterations,
            individual_ofv=base_results.individual_ofv,
            individual_estimates=base_results.individual_estimates,
            individual_estimates_covariance=base_results.individual_estimates_covariance,
            residuals=base_results.residuals,
            predictions=base_results.predictions,
            runtime_total=base_results.runtime_total,
            termination_cause=base_results.termination_cause,
            termination_cause_iterations=base_results.termination_cause,
            function_evaluations=base_results.function_evaluations,
            function_evaluations_iterations=base_results.function_evaluations_iterations,
            significant_digits=base_results.significant_digits,
            significant_digits_iterations=base_results.significant_digits_iterations,
            log_likelihood=base_results.log_likelihood,
            log=base_results.log,
            evaluation=base_results.evaluation,
        )
    )


def create_base_model(ss, model_or_model_entry):
    if isinstance(model_or_model_entry, ModelEntry):
        model = model_or_model_entry.model
        res = model_or_model_entry.modelfit_results
    else:
        model = model_or_model_entry
        res = None

    base = set_initial_estimates(model, res.parameter_estimates) if res else model

    model_mfl = get_model_features(model, supress_warnings=True)
    model_mfl = ModelFeatures.create_from_mfl_string(model_mfl)
    added_features = ""
    lnt = model_mfl.least_number_of_transformations(ss, tool="modelsearch")
    for name, func in lnt.items():
        base = func(base)
        added_features += f';{name[0]}({name[1]})'
    # UPDATE_DESCRIPTION
    # FIXME : Need to be its own parent if the input model shouldn't be ranked with the others
    base = base.replace(name="BASE", description=added_features[1:])

    return ModelEntry.create(base, modelfit_results=None, parent=None)


def post_process(rank_type, cutoff, strictness, context, *model_entries):
    res_model_entries = []
    input_model_entry = None
    base_model_entry = None
    for model_entry in model_entries:
        model = model_entry.model
        if not model.name.startswith('modelsearch_run') and model.name == "BASE":
            input_model_entry = model_entry
            base_model_entry = model_entry
        elif not model.name.startswith('modelsearch_run') and model.name != "BASE":
            user_input_model_entry = model_entry
        else:
            res_model_entries.append(model_entry)
    if not base_model_entry:
        input_model_entry = user_input_model_entry
        base_model_entry = user_input_model_entry
    if not input_model_entry:
        raise ValueError('Error in workflow: No input model')

    entries_to_summarize = [user_input_model_entry]

    if user_input_model_entry != base_model_entry:
        entries_to_summarize.append(base_model_entry)

    if res_model_entries:
        entries_to_summarize += res_model_entries

    summary_models = summarize_modelfit_results_from_entries(entries_to_summarize)
    summary_models['step'] = [0] + [1] * (len(summary_models) - 1)
    summary_models = summary_models.reset_index().set_index(['step', 'model'])

    res = create_results(
        ModelSearchResults,
        input_model_entry,
        base_model_entry,
        res_model_entries,
        rank_type,
        cutoff,
        summary_models=summary_models,
        strictness=strictness,
        context=context,
    )
    return res


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(
    search_space,
    algorithm,
    iiv_strategy,
    rank_type,
    model,
    strictness,
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

    if strictness is not None and "rse" in strictness.lower():
        if model.execution_steps[-1].parameter_uncertainty_method is None:
            raise ValueError(
                'parameter_uncertainty_method not set for model, cannot calculate relative standard errors.'
            )


@dataclass(frozen=True)
class ModelSearchResults(ToolResults):
    rst_path = Path(__file__).resolve().parent / 'report.rst'
    pass
