import pandas as pd

import pharmpy.results
import pharmpy.tools.iivsearch.algorithms as algorithms
from pharmpy.modeling import (
    add_pk_iiv,
    copy_model,
    create_joint_distribution,
    summarize_modelfit_results,
)
from pharmpy.tools.common import summarize_tool, summarize_tool_individuals
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow, call_workflow


def create_workflow(
    algorithm,
    iiv_strategy=0,
    rankfunc='bic',
    cutoff=None,
    model=None,
):
    """Run IIVsearch tool. For more details, see :ref:`iivsearch`.

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
        IIVsearch tool result object

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


def create_algorithm_workflow(input_model, base_model, algorithm, iiv_strategy, rankfunc, cutoff):
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
        input_model,
    )

    wf.add_task(task_result, predecessors=start_model_task + wf.output_tasks)

    return wf


def start(input_model, algorithm, iiv_strategy, rankfunc, cutoff):
    if iiv_strategy != 0:
        model_iiv = copy_model(input_model, 'base_model')
        _add_iiv(iiv_strategy, model_iiv)
        base_model = model_iiv
    else:
        base_model = input_model

    if algorithm == 'brute_force':
        list_of_algorithms = ['brute_force_no_of_etas', 'brute_force_block_structure']
    else:
        list_of_algorithms = [algorithm]

    sum_tools, sum_models, sum_inds, sum_inds_count = [], [], [], []
    for i, algorithm_cur in enumerate(list_of_algorithms):
        wf = create_algorithm_workflow(
            input_model, base_model, algorithm_cur, iiv_strategy, rankfunc, cutoff
        )
        next_res = call_workflow(wf, f'results_{algorithm}')
        if i == 0:
            res = next_res
        else:
            prev_models = [model.name for model in res.models]
            new_models = [model for model in next_res.models if model.name not in prev_models]
            res.models = res.models + new_models
            res.best_model = next_res.best_model
            res.input_model = input_model
        sum_tools.append(next_res.summary_tool)
        sum_models.append(next_res.summary_models)
        sum_inds.append(next_res.summary_individuals)
        sum_inds_count.append(next_res.summary_individuals_count)
        base_model = res.best_model
        iiv_strategy = 0

    if len(list_of_algorithms) > 1:
        keys = list(range(1, len(list_of_algorithms) + 1))
    else:
        keys = None

    res.summary_tool = _concat_summaries(sum_tools, keys)
    res.summary_models = _concat_summaries(sum_models, keys)
    res.summary_individuals = _concat_summaries(sum_inds, keys)
    res.summary_individuals_count = _concat_summaries(sum_inds_count, keys)

    return res


def _concat_summaries(summaries, keys):
    if keys:
        return pd.concat(summaries, keys=keys)
    else:
        return pd.concat(summaries)


def _results(res):
    return res


def _start_algorithm(model):
    model.parent_model = model.name
    return model


def _add_iiv(iiv_strategy, model):
    add_pk_iiv(model)
    if iiv_strategy == 2:
        create_joint_distribution(model)
    return model


def post_process_results(rankfunc, cutoff, input_model, *models):
    base_model, res_models = models

    if isinstance(res_models, tuple):
        res_models = list(res_models)
    else:
        res_models = [res_models]

    summary_tool = summarize_tool(
        res_models,
        base_model,
        rankfunc,
        cutoff,
        bic_type='iiv',
    )
    summary_models = summarize_modelfit_results([base_model] + res_models).sort_values(
        by=[rankfunc]
    )
    summary_individuals, summary_individuals_count = summarize_tool_individuals(
        [base_model] + res_models, summary_tool['description'], summary_tool[f'd{rankfunc}']
    )

    best_model_name = summary_tool['rank'].idxmin()
    try:
        best_model = [model for model in res_models if model.name == best_model_name][0]
    except IndexError:
        best_model = base_model

    if base_model.name != input_model.name:
        res_models.insert(0, base_model)

    res = IIVResults(
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_individuals=summary_individuals,
        summary_individuals_count=summary_individuals_count,
        best_model=best_model,
        input_model=input_model,
        models=res_models,
    )

    return res


class IIVResults(pharmpy.results.Results):
    def __init__(
        self,
        summary_tool=None,
        summary_models=None,
        summary_individuals=None,
        summary_individuals_count=None,
        best_model=None,
        input_model=None,
        models=None,
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals = summary_individuals
        self.summary_individuals_count = summary_individuals_count
        self.best_model = best_model
        self.input_model = input_model
        self.models = models
