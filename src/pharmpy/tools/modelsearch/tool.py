import numpy as np
import pandas as pd

import pharmpy.execute as execute
import pharmpy.results
import pharmpy.tools
import pharmpy.tools.modelfit as modelfit
import pharmpy.tools.modelsearch.algorithms as algorithms
import pharmpy.tools.modelsearch.rankfuncs as rankfuncs
from pharmpy.tools.modelfit import create_single_fit_workflow
from pharmpy.tools.workflows import Task, Workflow


class ModelSearch(pharmpy.tools.Tool):
    def __init__(self, start_model, algorithm, mfl, rankfunc='ofv', cutoff=None, **kwargs):
        self.start_model = start_model
        self.mfl = mfl
        self.algorithm = getattr(algorithms, algorithm)
        self.rankfunc = getattr(rankfuncs, rankfunc)
        self.cutoff = cutoff
        super().__init__(**kwargs)
        self.start_model.database = self.database.model_database

    def fit(self, models):
        db = execute.LocalDirectoryDatabase(self.rundir.path / 'models')
        modelfit_run = modelfit.Modelfit(models, database=db, path=self.rundir.path)
        modelfit_run.run()

    def run(self):
        df = self.algorithm(
            self.start_model,
            self.mfl,
            self.fit,
            self.rankfunc,
        )
        res = ModelSearchResults(summary=df)
        res.to_json(path=self.rundir.path / 'results.json')
        res.to_csv(path=self.rundir.path / 'results.csv')
        return res


def create_workflow(model, algorithm, mfl, rankfunc='ofv', cutoff=None):
    algorithm_func = getattr(algorithms, algorithm)
    rankfunc_func = getattr(rankfuncs, rankfunc)

    wf = Workflow()
    wf.name = 'modelsearch'

    start_task = Task('start_modelsearch', start, model)
    wf.add_task(start_task)

    start_model_task = []
    if not model.modelfit_results:
        wf_fit = create_single_fit_workflow()
        wf.insert_workflow(wf_fit, predecessors=start_task)
        start_model_task = wf_fit.output_tasks

    wf_search, candidate_model_tasks, model_features = algorithm_func(mfl)
    wf.insert_workflow(wf_search, predecessors=wf.output_tasks)

    task_result = Task(
        'results',
        post_process_results,
        model,
        rankfunc_func,
        cutoff,
        model_features,
    )

    wf.add_task(task_result, predecessors=start_model_task + candidate_model_tasks)

    return wf


def start(model):
    return model


def post_process_results(start_model, rankfunc, cutoff, model_features, *models):
    res_data = {'dofv': [], 'features': [], 'rank': []}
    model_names = []

    res_models = []
    for model in models:
        model.modelfit_results.estimation_step
        if model.name == start_model.name:
            start_model.modelfit_results = model.modelfit_results
        else:
            res_models.append(model)

    if cutoff is not None:
        ranks = rankfunc(start_model, res_models, cutoff=cutoff)
    else:
        ranks = rankfunc(start_model, res_models)

    for model in res_models:
        model_names.append(model.name)
        res_data['dofv'].append(start_model.modelfit_results.ofv - model.modelfit_results.ofv)
        res_data['features'].append(model_features[model.name])
        if model in ranks:
            res_data['rank'].append(ranks.index(model) + 1)
        else:
            res_data['rank'].append(np.nan)

    # FIXME: in ranks, if any row has NaN the rank converts to float
    df = pd.DataFrame(res_data, index=model_names)

    best_model_name = df['rank'].idxmin()
    try:
        best_model = [model for model in res_models if model.name == best_model_name][0]
    except IndexError:
        best_model = None

    res = ModelSearchResults(
        summary=df, best_model=best_model, start_model=start_model, models=res_models
    )

    return res


class ModelSearchResults(pharmpy.results.Results):
    def __init__(self, summary=None, best_model=None, start_model=None, models=None):
        self.summary = summary
        self.best_model = best_model
        self.start_model = start_model
        self.models = models


def run_modelsearch(base_model, algorithm, mfl, **kwargs):
    ms = ModelSearch(base_model, algorithm, mfl, **kwargs)
    res = ms.run()
    return res
