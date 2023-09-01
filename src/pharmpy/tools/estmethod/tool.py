from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import pharmpy.tools.estmethod.algorithms as algorithms
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import has_linear_odes
from pharmpy.results import ModelfitResults
from pharmpy.tools import summarize_errors, summarize_modelfit_results
from pharmpy.tools.common import ToolResults
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow, WorkflowBuilder

EST_METHODS = ('FOCE', 'FO', 'IMP', 'IMPMAP', 'ITS', 'SAEM', 'LAPLACE', 'BAYES')
SOLVERS = ('CVODES', 'DGEAR', 'DVERK', 'IDA', 'LSODA', 'LSODI')
UNCERT_METHODS = ('SANDWICH', 'CPG', 'OFIM')

ALGORITHMS = frozenset(['exhaustive', 'exhaustive_with_update', 'exhaustive_only_eval'])


def create_workflow(
    algorithm: str,
    est_methods: Optional[Union[List[str], str]] = None,
    solvers: Optional[Union[List[str], str]] = None,
    uncert_methods: Optional[Union[List[str], str]] = None,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
):
    """Run estmethod tool.

     Parameters
     ----------
     algorithm : str
         The algorithm to use (can be 'exhaustive', 'exhaustive_with_update' or 'exhaustive_only_eval')
    est_ methods : list or None
         List of estimation methods to test.
         Can be specified as 'all', a list of estimation methods, or None (to not test any estimation method)
     solvers : list, str or None
         List of solver to test. Can be specified as 'all', a list of solvers, or None (to
         not test any solver)
     uncert_method : list, str or None
         List of parameter uncertainty methods to test.
         Can be specified as 'all', a list of uncertainty methods, or None (to not evaluate any uncertainty)
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
     >>> est_methods = ['imp', 'saem']
     >>> uncert_methods = None
     >>> run_estmethod( # doctest: +SKIP
     >>>     'reduced', est_methods=est_methods, solvers='all', # doctest: +SKIP
     >>>      uncert_methods=uncert_methods, results=results, model=model # doctest: +SKIP
     >>> ) # doctest: +SKIP

    """
    wb = WorkflowBuilder(name="estmethod")

    algorithm_func = getattr(algorithms, algorithm)

    if model is not None:
        start_task = Task('start_estmethod', start, model)
    else:
        start_task = Task('start_estmethod', start)

    wb.add_task(start_task)

    if est_methods is None:
        est_methods = [model.estimation_steps[-1].est_method]

    wf_algorithm, task_base_model_fit = algorithm_func(
        _format_input(est_methods, EST_METHODS),
        _format_input(solvers, SOLVERS),
        _format_input(uncert_methods, UNCERT_METHODS),
    )
    wb.insert_workflow(wf_algorithm, predecessors=start_task)

    wf_fit = create_fit_workflow(n=len(wb.output_tasks))
    wb.insert_workflow(wf_fit, predecessors=wb.output_tasks)

    if task_base_model_fit:
        model_tasks = wb.output_tasks + task_base_model_fit
    else:
        model_tasks = wb.output_tasks

    task_post_process = Task('post_process', post_process, model)
    wb.add_task(task_post_process, predecessors=model_tasks)

    return Workflow(wb)


def _format_input(input_option, default_option):
    if input_option == 'all':
        return default_option
    elif input_option is None:
        return [None]
    else:
        return [entry.upper() for entry in input_option]


def start(model):
    return model


def post_process(input_model, *models):
    res_models = []
    base_model = input_model
    for model in models:
        if model.name == 'base_model':
            base_model = model
        else:
            res_models.append(model)

    summary_tool = summarize_tool(models)
    summary_models = summarize_modelfit_results(
        [base_model.modelfit_results] + [model.modelfit_results for model in res_models],
        include_all_estimation_steps=True,
    )
    summary_errors = summarize_errors(m.modelfit_results for m in models)
    summary_settings = summarize_estimation_steps([base_model] + res_models)

    if base_model.name == input_model.name:
        models = res_models
    else:
        models = [base_model] + res_models

    return EstMethodResults(
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_settings=summary_settings,
        summary_errors=summary_errors,
        models=models,
    )


@dataclass(frozen=True)
class EstMethodResults(ToolResults):
    rst_path = Path(__file__).resolve().parent / 'report.rst'

    summary_settings: Optional[Any] = None


def summarize_tool(models):
    rows = {}

    for model in models:
        description, parent_model = model.description, model.parent_model
        res = model.modelfit_results
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


def summarize_estimation_steps(models):
    dfs = {}
    for model in models:
        df = model.estimation_steps.to_dataframe()
        df.index = range(1, len(df) + 1)
        dfs[model.name] = df.drop(columns=['tool_options'])

    return pd.concat(dfs.values(), keys=dfs.keys())


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(algorithm, est_methods, solvers, uncert_methods, model):
    if solvers is not None and has_linear_odes(model):
        raise ValueError(
            'Invalid input `model`: testing non-linear solvers on linear system is not supported'
        )

    if algorithm not in ALGORITHMS:
        raise ValueError(
            f'Invalid `algorithm`: got `{algorithm}`, must be one of {sorted(ALGORITHMS)}.'
        )

    if est_methods is None and solvers is None and uncert_methods is None:
        raise ValueError(
            'Invalid search space options: please specify at least one of `est_methods`, `solvers`, or `uncert_methods`'
        )

    if est_methods is not None:
        _validate_search_space(est_methods, EST_METHODS, 'est_methods')

    if solvers is not None:
        _validate_search_space(solvers, SOLVERS, 'solvers')

    if uncert_methods is not None:
        _validate_search_space(uncert_methods, UNCERT_METHODS, 'uncert_methods')


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
