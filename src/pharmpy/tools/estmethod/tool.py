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
from pharmpy.workflows import Task, Workflow

EST_METHODS = ('FOCE', 'FO', 'IMP', 'IMPMAP', 'ITS', 'SAEM', 'LAPLACE', 'BAYES')
SOLVERS = ('CVODES', 'DGEAR', 'DVERK', 'IDA', 'LSODA', 'LSODI')
COVS = ('SANDWICH', 'CPG', 'OFIM')

ALGORITHMS = frozenset(['exhaustive', 'exhaustive_with_update', 'exhaustive_only_eval'])


def create_workflow(
    algorithm: str,
    methods: Optional[Union[List[str], str]] = None,
    solvers: Optional[Union[List[str], str]] = None,
    covs: Optional[Union[List[str], str]] = None,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
):
    """Run estmethod tool.

    Parameters
    ----------
    algorithm : str
        The algorithm to use (can be 'exhaustive', 'exhaustive_with_update' or 'exhaustive_only_eval')
    methods : list or None
        List of estimation methods to test. Can be specified as 'all', a list of methods, or
        None (to not test any estimation method)
    solvers : list, str or None
        List of solver to test. Can be specified as 'all', a list of solvers, or None (to
        not test any solver)
    covs : list, str or None
        List of covariance to test. Can be specified as 'all', a list of covs, or None (to
        not evaluate any covariance)
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
    >>> methods = ['imp', 'saem']
    >>> covs = None
    >>> run_estmethod( # doctest: +SKIP
    >>>     'reduced', methods=methods, solvers='all', covs=covs, results=results, model=model # doctest: +SKIP
    >>> ) # doctest: +SKIP

    """
    wf = Workflow()
    wf.name = "estmethod"

    algorithm_func = getattr(algorithms, algorithm)

    if model is not None:
        start_task = Task('start_estmethod', start, model)
    else:
        start_task = Task('start_estmethod', start)

    wf.add_task(start_task)

    if methods is None:
        methods = [model.estimation_steps[-1].method]

    wf_algorithm, task_base_model_fit = algorithm_func(
        _format_input(methods, EST_METHODS),
        _format_input(solvers, SOLVERS),
        _format_input(covs, COVS),
    )
    wf.insert_workflow(wf_algorithm, predecessors=start_task)

    wf_fit = create_fit_workflow(n=len(wf.output_tasks))
    wf.insert_workflow(wf_fit, predecessors=wf.output_tasks)

    if task_base_model_fit:
        model_tasks = wf.output_tasks + task_base_model_fit
    else:
        model_tasks = wf.output_tasks

    task_post_process = Task('post_process', post_process, model)
    wf.add_task(task_post_process, predecessors=model_tasks)

    return wf


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
def validate_input(algorithm, methods, solvers, covs, model):
    if solvers is not None and has_linear_odes(model):
        raise ValueError(
            'Invalid input `model`: testing non-linear solvers on linear system is not supported'
        )

    if algorithm not in ALGORITHMS:
        raise ValueError(
            f'Invalid `algorithm`: got `{algorithm}`, must be one of {sorted(ALGORITHMS)}.'
        )

    if methods is None and solvers is None and covs is None:
        raise ValueError(
            'Invalid search space options: please specify at least one of `methods`, `solvers`, or `covs`'
        )

    if methods is not None:
        _validate_search_space(methods, EST_METHODS, 'methods')

    if solvers is not None:
        _validate_search_space(solvers, SOLVERS, 'solvers')

    if covs is not None:
        _validate_search_space(covs, COVS, 'covs')


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
