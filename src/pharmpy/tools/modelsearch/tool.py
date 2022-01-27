import numpy as np
import pandas as pd

import pharmpy.results
import pharmpy.tools.modelsearch.algorithms as algorithms
import pharmpy.tools.modelsearch.rankfuncs as rankfuncs
from pharmpy.modeling import summarize_modelfit_results
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow


def create_workflow(
    algorithm, mfl, rankfunc='ofv', cutoff=None, add_etas=False, etas_as_fullblock=False, model=None
):
    algorithm_func = getattr(algorithms, algorithm)

    wf = Workflow()
    wf.name = 'modelsearch'

    if model:
        start_task = Task('start_modelsearch', start, model)
    else:
        start_task = Task('start_modelsearch', start)

    wf.add_task(start_task)

    if model and not model.modelfit_results:
        wf_fit = create_fit_workflow(n=1)
        wf.insert_workflow(wf_fit, predecessors=start_task)
        start_model_task = wf_fit.output_tasks
    else:
        start_model_task = [start_task]

    wf_search, candidate_model_tasks, model_features = algorithm_func(
        mfl, add_etas, etas_as_fullblock
    )
    wf.insert_workflow(wf_search, predecessors=wf.output_tasks)

    task_result = Task(
        'results',
        post_process_results,
        rankfunc,
        cutoff,
        model_features,
    )

    wf.add_task(task_result, predecessors=start_model_task + candidate_model_tasks)

    return wf


def start(model):
    return model


def post_process_results(rankfunc, cutoff, model_features, *models):
    res_models = []
    for model in models:
        if not model.name.startswith('modelsearch_candidate'):
            start_model = model
        else:
            res_models.append(model)

    summary_tool = create_summary(res_models, start_model, rankfunc, cutoff, model_features)

    best_model_name = summary_tool['rank'].idxmin()
    try:
        best_model = [model for model in res_models if model.name == best_model_name][0]
    except IndexError:
        best_model = start_model

    summary_models = summarize_modelfit_results([start_model] + res_models)

    res = ModelSearchResults(
        summary_tool=summary_tool,
        summary_models=summary_models,
        best_model=best_model,
        start_model=start_model,
        models=res_models,
    )

    return res


class ModelSearchResults(pharmpy.results.Results):
    def __init__(
        self, summary_tool=None, summary_models=None, best_model=None, start_model=None, models=None
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.best_model = best_model
        self.start_model = start_model
        self.models = models


def create_summary(models, start_model, rankfunc, cutoff, model_features):
    rankfunc = getattr(rankfuncs, rankfunc)

    res_data = {'parent_model': [], 'dofv': [], 'features': [], 'rank': []}
    model_names = []

    if cutoff is not None:
        ranks = rankfunc(start_model, models, cutoff=cutoff)
    else:
        ranks = rankfunc(start_model, models)

    for model in models:
        model_names.append(model.name)
        res_data['parent_model'].append(model.parent_model)
        try:
            res_data['dofv'].append(start_model.modelfit_results.ofv - model.modelfit_results.ofv)
        except AttributeError:
            res_data['dofv'].append(np.nan)
        res_data['features'].append(model_features[model.name])
        if model in ranks:
            res_data['rank'].append(ranks.index(model) + 1)
        else:
            res_data['rank'].append(np.nan)

    # FIXME: in ranks, if any row has NaN the rank converts to float
    return pd.DataFrame(res_data, index=model_names)
