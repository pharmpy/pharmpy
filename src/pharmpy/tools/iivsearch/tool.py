import pandas as pd

import pharmpy.results
import pharmpy.tools.iivsearch.algorithms as algorithms
from pharmpy.modeling import (
    add_pk_iiv,
    copy_model,
    create_joint_distribution,
    summarize_modelfit_results,
)
from pharmpy.tools.common import summarize_tool
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow, call_workflow


def create_workflow(
    algorithm,
    iiv_strategy=0,
    rankfunc='bic',
    cutoff=None,
    model=None,
):
    """Run IIVSearch tool.

    Parameters
    ----------
    algorithm : str
        Which algorithm to run (brute_force, brute_force_no_of_etas, brute_force_block_structure)
    iiv_strategy : int
        How IIVs should be added to start model. Default is 0 (no added IIVs)
    rankfunc : str
        Which ranking function should be used (OFV, AIC, BIC). Default is BIC
    cutoff : float
        Cutoff for which value of the ranking function that is considered significant. Default
        is None (all models will be ranked)
    model : Model
        Pharmpy model

    Returns
    -------
    IIVResults
        IIVSearch tool result object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> run_iivsearch('brute_force', model=model)      # doctest: +SKIP

    """
    wf = Workflow()
    wf.name = 'iivsearch'
    start_task = Task('start_iiv', start, model, algorithm, iiv_strategy, rankfunc, cutoff)
    wf.add_task(start_task)
    task_results = Task('results', _results)
    wf.add_task(task_results, predecessors=[start_task])
    return wf


def create_algorithm_workflow(base_model, algorithm, iiv_strategy, rankfunc, cutoff):
    wf = Workflow()

    start_task = Task(f'start_{algorithm}', _start_algorithm, base_model)
    wf.add_task(start_task)

    if iiv_strategy != 0:
        wf_fit = create_fit_workflow(n=1)
        wf.insert_workflow(wf_fit)
        start_model_task = wf_fit.output_tasks
    else:
        start_model_task = [start_task]

    algorithm_func = getattr(algorithms, algorithm)
    wf_method = algorithm_func(base_model)
    wf.insert_workflow(wf_method)

    task_result = Task(
        'results',
        post_process_results,
        rankfunc,
        cutoff,
    )

    wf.add_task(task_result, predecessors=start_model_task + wf.output_tasks)

    return wf


def start(model, algorithm, iiv_strategy, rankfunc, cutoff):
    if iiv_strategy != 0:
        model_iiv = copy_model(model, f'{model.name}_add_iiv')
        _add_iiv(iiv_strategy, model_iiv)
        base_model = model_iiv
    else:
        base_model = model

    if algorithm == 'brute_force':
        list_of_algorithms = ['brute_force_no_of_etas', 'brute_force_block_structure']
    else:
        list_of_algorithms = [algorithm]

    for i, algorithm_cur in enumerate(list_of_algorithms):
        wf = create_algorithm_workflow(base_model, algorithm_cur, iiv_strategy, rankfunc, cutoff)
        next_res = call_workflow(wf, f'results{algorithm}')
        if i == 0:
            res = next_res
        else:
            res.models = res.models + next_res.models
            res.best_model = next_res.best_model
            res.summary_tool = pd.concat([res.summary_tool, next_res.summary_tool])
            res.summary_models = pd.concat([res.summary_models, next_res.summary_models])
        base_model = res.best_model
        iiv_strategy = 0

    return res


def _results(res):
    return res


def _start_algorithm(model):
    return model


def _add_iiv(iiv_strategy, model):
    add_pk_iiv(model)
    if iiv_strategy == 2:
        create_joint_distribution(model)
    return model


def post_process_results(rankfunc, cutoff, *models):
    start_model, res_models = models

    if isinstance(res_models, tuple):
        res_models = list(res_models)
    else:
        res_models = [res_models]

    summary_tool = summarize_tool(
        res_models,
        start_model,
        rankfunc,
        cutoff,
        bic_type='iiv',
    )
    summary_models = summarize_modelfit_results([start_model] + res_models)

    best_model_name = summary_tool['rank'].idxmin()
    try:
        best_model = [model for model in res_models if model.name == best_model_name][0]
    except IndexError:
        best_model = start_model

    res = IIVResults(
        summary_tool=summary_tool,
        summary_models=summary_models,
        best_model=best_model,
        start_model=start_model,
        models=res_models,
    )

    return res


class IIVResults(pharmpy.results.Results):
    def __init__(
        self, summary_tool=None, summary_models=None, best_model=None, start_model=None, models=None
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.best_model = best_model
        self.start_model = start_model
        self.models = models
