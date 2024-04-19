import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import pharmpy.tools.estmethod.algorithms as algorithms
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import has_linear_odes
from pharmpy.tools.common import ToolResults
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import summarize_errors_from_entries, summarize_modelfit_results_from_entries
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults

METHODS = ('FOCE', 'FO', 'IMP', 'IMPMAP', 'ITS', 'SAEM', 'LAPLACE', 'BAYES')
SOLVERS = ('CVODES', 'DGEAR', 'DVERK', 'IDA', 'LSODA', 'LSODI')
PARAMETER_UNCERTAINTY_METHODS = ('SANDWICH', 'SMAT', 'RMAT')

ALGORITHMS = frozenset(['exhaustive', 'exhaustive_with_update', 'exhaustive_only_eval'])


def create_workflow(
    algorithm: Literal[tuple(ALGORITHMS)],
    methods: Optional[Union[List[Literal[METHODS]], Literal['all']]] = None,
    solvers: Optional[Union[List[Literal[SOLVERS]], Literal[SOLVERS]]] = None,
    parameter_uncertainty_methods: Optional[
        Union[List[Literal[PARAMETER_UNCERTAINTY_METHODS]], Literal[PARAMETER_UNCERTAINTY_METHODS]]
    ] = None,
    compare_ofv: bool = True,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
):
    """Run estmethod tool.

    Parameters
    ----------
    algorithm : str
         The algorithm to use (can be 'exhaustive', 'exhaustive_with_update' or 'exhaustive_only_eval')
    methods : list of {'FOCE', 'FO', 'IMP', 'IMPMAP', 'ITS', 'SAEM', 'LAPLACE', 'BAYES'}, None or 'all'
         List of estimation methods to test.
         Can be specified as 'all', a list of estimation methods, or None (to not test any estimation method)
    solvers : str or list of {'CVODES', 'DGEAR', 'DVERK', 'IDA', 'LSODA', 'LSODI'} or None
         List of solver to test. Can be specified as 'all', a list of solvers, or None (to
         not test any solver)
    parameter_uncertainty_methods : str or list of {'SANDWICH', 'SMAT', 'RMAT'} or None
         List of parameter uncertainty methods to test.
         Can be specified as 'all', a list of uncertainty methods, or None (to not evaluate any uncertainty)
    compare_ofv : bool
        Whether to compare the OFV between candidates. Comparison is made by evaluating using IMP
    results : ModelfitResults
         Results for model
    model : Model
         Pharmpy model

    Returns
    -------
    EstMethodResults
        Estmethod tool result object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> from pharmpy.tools import run_estmethod, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> methods = ['IMP', 'SAEM']
    >>> parameter_uncertainty_methods = None
    >>> run_estmethod( # doctest: +SKIP
    >>>     'reduced', methods=methods, solvers='all', # doctest: +SKIP
    >>>      parameter_uncertainty_methods=parameter_uncertainty_methods, results=results, model=model # doctest: +SKIP
    >>> ) # doctest: +SKIP

    """
    wb = WorkflowBuilder(name="estmethod")

    algorithm_func = getattr(algorithms, algorithm)

    if model is not None:
        start_task = Task('start_estmethod', start, model, results)
    else:
        start_task = Task('start_estmethod', start)

    wb.add_task(start_task)

    if methods is None:
        methods = [model.execution_steps[-1].method]

    args = [
        _format_input(methods, METHODS),
        _format_input(solvers, SOLVERS),
        _format_input(parameter_uncertainty_methods, PARAMETER_UNCERTAINTY_METHODS),
    ]

    if algorithm != 'exhaustive_only_eval':
        args.append(compare_ofv)

    wf_algorithm, task_base_model_fit = algorithm_func(*args)
    wb.insert_workflow(wf_algorithm, predecessors=start_task)

    wf_fit = create_fit_workflow(n=len(wb.output_tasks))
    wb.insert_workflow(wf_fit, predecessors=wb.output_tasks)

    model_tasks = [start_task] + wb.output_tasks
    if task_base_model_fit:
        model_tasks.extend(task_base_model_fit)

    task_post_process = Task('post_process', post_process)
    wb.add_task(task_post_process, predecessors=model_tasks)

    return Workflow(wb)


def _format_input(input_option, default_option):
    if input_option == 'all':
        return default_option
    elif input_option is None:
        return [None]
    else:
        return [entry.upper() for entry in input_option]


def start(model, modelfit_results):
    log = modelfit_results.log if modelfit_results else None
    model_entry = ModelEntry(model, modelfit_results=modelfit_results, log=log)
    return model_entry


def post_process(*model_entries):
    summary_tool = summarize_tool(model_entries)

    res_models = [model_entry.model for model_entry in model_entries]

    summary_models = summarize_modelfit_results_from_entries(
        model_entries,
        include_all_execution_steps=True,
    )
    summary_errors = summarize_errors_from_entries(model_entries)
    summary_settings = summarize_execution_steps(res_models)

    return EstMethodResults(
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_settings=summary_settings,
        summary_errors=summary_errors,
    )


@dataclass(frozen=True)
class EstMethodResults(ToolResults):
    rst_path = Path(__file__).resolve().parent / 'report.rst'

    summary_settings: Optional[Any] = None


def summarize_tool(model_entries):
    rows = {}

    for model_entry in model_entries:
        model, res = model_entry.model, model_entry.modelfit_results
        description, parent_model = model.description, model.parent_model
        if res is not None:
            ofv = res.ofv
            runtime_est = res.estimation_runtime_iterations.iloc[0]
        else:
            ofv, runtime_est = np.nan, np.nan
        rows[model.name] = (description, ofv, runtime_est, parent_model)

    colnames = ['description', 'ofv', 'runtime_estimation', 'parent_model']
    index = pd.Index(rows.keys(), name='model')
    df = pd.DataFrame(rows.values(), index=index, columns=colnames)

    return df.sort_values(by=['ofv'])


def summarize_execution_steps(models):
    dfs = {}
    for model in models:
        df = model.execution_steps.to_dataframe()
        df.index = range(1, len(df) + 1)
        dfs[model.name] = df.drop(columns=['tool_options'])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA",
            category=FutureWarning,
        )
        summary = pd.concat(list(dfs.values()), keys=list(dfs.keys()))
    return summary


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(algorithm, methods, solvers, parameter_uncertainty_methods, model):
    if solvers is not None and has_linear_odes(model):
        raise ValueError(
            'Invalid input `model`: testing non-linear solvers on linear system is not supported'
        )

    if methods is None and solvers is None and parameter_uncertainty_methods is None:
        raise ValueError(
            'Invalid search space options: please specify at least '
            'one of `methods`, `solvers`, or `parameter_uncertainty_methods`'
        )

    if methods is not None:
        _validate_search_space(methods, METHODS, 'methods')

    if solvers is not None:
        _validate_search_space(solvers, SOLVERS, 'solvers')

    if parameter_uncertainty_methods is not None:
        _validate_search_space(
            parameter_uncertainty_methods,
            PARAMETER_UNCERTAINTY_METHODS,
            'parameter_uncertainty_methods',
        )


def _validate_search_space(input_search_space, allowed_search_space, option_name):
    if isinstance(input_search_space, str):
        if input_search_space != 'all':
            raise ValueError(
                f'Invalid `{option_name}`: if option is str it must be `all`, got {input_search_space}'
            )
    else:
        option_diff = {option.upper() for option in input_search_space}.difference(
            allowed_search_space
        )
        if option_diff:
            raise ValueError(
                f'Invalid `{option_name}`: {option_diff} not in {sorted(allowed_search_space)}'
            )
