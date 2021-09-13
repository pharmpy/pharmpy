import numpy as np
import pandas as pd

import pharmpy.execute as execute
import pharmpy.results
import pharmpy.tools
import pharmpy.tools.modelfit as modelfit
import pharmpy.tools.modelsearch.algorithms as algorithms
import pharmpy.tools.modelsearch.rankfuncs as rankfuncs
from pharmpy.tools.workflows import Task


class ModelSearch(pharmpy.tools.Tool):
    def __init__(self, base_model, algorithm, mfl, rankfunc='ofv', cutoff=None, **kwargs):
        self.base_model = base_model
        self.mfl = mfl
        self.algorithm = getattr(algorithms, algorithm)
        self.rankfunc = getattr(rankfuncs, rankfunc)
        self.cutoff = cutoff
        super().__init__(**kwargs)
        self.base_model.database = self.database.model_database

    def fit(self, models):
        db = execute.LocalDirectoryDatabase(self.rundir.path / 'models')
        modelfit_run = modelfit.Modelfit(models, database=db, path=self.rundir.path)
        modelfit_run.run()

    def run(self):
        if self.algorithm.__name__ == 'exhaustive_stepwise':
            wf, model_tasks, model_features = self.algorithm(self.base_model, self.mfl)

            task_result = Task(
                'results',
                post_process_results,
                self.base_model,
                self.rankfunc,
                self.cutoff,
                model_features,
            )
            wf.add_task(task_result, predecessors=model_tasks)

            res = self.dispatcher.run(wf, self.database)
            self.base_model.modelfit_results = res.base_model.modelfit_results
            return res
        else:
            df = self.algorithm(
                self.base_model,
                self.mfl,
                self.fit,
                self.rankfunc,
            )
            res = ModelSearchResults(summary=df)
            res.to_json(path=self.rundir.path / 'results.json')
            res.to_csv(path=self.rundir.path / 'results.csv')
        return res


def post_process_results(base_model, rankfunc, cutoff, model_features, *models):
    res_data = {'dofv': [], 'features': [], 'rank': []}
    model_names = []

    res_models = []
    for model in models:
        model.modelfit_results.estimation_step
        if model.name == base_model.name:
            base_model.modelfit_results = model.modelfit_results
        else:
            res_models.append(model)

    if cutoff is not None:
        ranks = rankfunc(base_model, res_models, cutoff=cutoff)
    else:
        ranks = rankfunc(base_model, res_models)

    for model in res_models:
        model_names.append(model.name)
        res_data['dofv'].append(base_model.modelfit_results.ofv - model.modelfit_results.ofv)
        res_data['features'].append(model_features[model.name])
        if model in ranks:
            res_data['rank'].append(ranks.index(model) + 1)
        else:
            res_data['rank'].append(np.nan)

    # FIXME: in ranks, if any row has NaN the rank converts to float
    df = pd.DataFrame(res_data, index=model_names)

    best_model_name = df['rank'].idxmin()
    best_model = [model for model in res_models if model.name == best_model_name][0]

    res = ModelSearchResults(
        summary=df, best_model=best_model, base_model=base_model, models=res_models
    )

    return res


class ModelSearchResults(pharmpy.results.Results):
    def __init__(self, summary=None, best_model=None, base_model=None, models=None):
        self.summary = summary
        self.best_model = best_model
        self.base_model = base_model
        self.models = models


def run_modelsearch(base_model, algorithm, mfl, **kwargs):
    ms = ModelSearch(base_model, algorithm, mfl, **kwargs)
    res = ms.run()
    return res
