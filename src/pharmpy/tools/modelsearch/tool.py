import numpy as np
import pandas as pd

import pharmpy.results
import pharmpy.tools.modelsearch.algorithms as algorithms
import pharmpy.tools.modelsearch.rankfuncs as rankfuncs
from pharmpy.modeling import summarize_individuals, summarize_modelfit_results
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow


def create_workflow(
    mfl,
    algorithm,
    iiv_strategy=0,
    rankfunc='ofv',
    cutoff=None,
    model=None,
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

    wf_search, candidate_model_tasks, model_features = algorithm_func(mfl, iiv_strategy)
    wf.insert_workflow(wf_search, predecessors=wf.output_tasks)

    task_result = Task(
        'results',
        post_process_results,
        algorithm,
        rankfunc,
        cutoff,
        model_features,
    )

    wf.add_task(task_result, predecessors=start_model_task + candidate_model_tasks)

    return wf


def start(model):
    return model


def post_process_results(algorithm, rankfunc, cutoff, model_features, *models):
    res_models = []
    start_model = None
    for model in models:
        if not model.name.startswith('modelsearch_candidate'):
            start_model = model
        else:
            res_models.append(model)

    if not start_model:
        raise ValueError('Error in workflow: No starting model')

    if algorithm == 'reduced_stepwise':
        model_features = _update_model_features(start_model, res_models, model_features)

    summary_tool = create_summary(res_models, start_model, rankfunc, cutoff, model_features)

    best_model_name = summary_tool['rank'].idxmin()
    try:
        best_model = [model for model in res_models if model.name == best_model_name][0]
    except IndexError:
        best_model = start_model

    summary_models = summarize_modelfit_results([start_model] + res_models)
    summary_individuals = summarize_individuals([start_model] + res_models)

    res = ModelSearchResults(
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_individuals=summary_individuals,
        best_model=best_model,
        start_model=start_model,
        models=res_models,
    )

    return res


def _update_model_features(start_model, res_models, model_features_original):
    model_features_new = dict()
    for model in res_models:
        if model.name == start_model.name:
            feat_all = None
        else:
            feat = model_features_original[model.name]
            if (
                model.parent_model in model_features_new.keys()
                and model.parent_model != start_model.name
            ):
                feat_parent = model_features_new[model.parent_model]
                feat_all = feat_parent + (feat,)
            else:
                feat_all = (feat,)
        model_features_new[model.name] = feat_all

    return model_features_new


class ModelSearchResults(pharmpy.results.Results):
    def __init__(
        self,
        summary_tool=None,
        summary_models=None,
        summary_individuals=None,
        best_model=None,
        start_model=None,
        models=None,
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals = summary_individuals
        self.best_model = best_model
        self.start_model = start_model
        self.models = models


def create_summary(
    models,
    start_model,
    rankfunc_name,
    cutoff,
    model_features,
    bic_type='mixed',
):
    models_all = [start_model] + models
    rankfunc = getattr(rankfuncs, rankfunc_name)

    kwargs = dict()
    if cutoff is not None:
        kwargs['cutoff'] = cutoff
    if rankfunc_name == 'bic':
        kwargs['bic_type'] = bic_type
    ranking, diff_dict = rankfunc(start_model, models_all, **kwargs)
    ranking_by_name = [model.name for model in ranking]  # Using list of models is very slow

    index = []
    rows = []
    for model in models_all:
        index.append(model.name)
        parent_model = model.parent_model
        diff = diff_dict[model.name]
        if model.name == start_model.name:
            feat = None
        else:
            feat = model_features[model.name]
        if model.name in ranking_by_name:
            ranks = ranking_by_name.index(model.name) + 1
        else:
            ranks = np.nan
        rows.append([parent_model, diff, feat, ranks])

    # FIXME: in ranks, if any row has NaN the rank converts to float
    colnames = ['parent_model', f'd{rankfunc_name}', 'features', 'rank']
    df = pd.DataFrame(rows, index=index, columns=colnames)

    return df
