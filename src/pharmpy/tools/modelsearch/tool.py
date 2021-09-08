import pharmpy.execute as execute
import pharmpy.results
import pharmpy.tools
import pharmpy.tools.modelfit as modelfit
import pharmpy.tools.modelsearch.algorithms as algorithms
import pharmpy.tools.modelsearch.rankfuncs as rankfuncs
from pharmpy.tools.workflows import Task


class ModelSearch(pharmpy.tools.Tool):
    def __init__(self, base_model, algorithm, mfl, rankfunc='ofv', **kwargs):
        self.base_model = base_model
        self.mfl = mfl
        self.algorithm = getattr(algorithms, algorithm)
        self.rankfunc = getattr(rankfuncs, rankfunc)
        super().__init__(**kwargs)
        self.base_model.database = self.database.model_database

    def fit(self, models):
        db = execute.LocalDirectoryDatabase(self.rundir.path / 'models')
        modelfit_run = modelfit.Modelfit(models, database=db, path=self.rundir.path)
        modelfit_run.run()

    def run(self):
        if self.algorithm.__name__ == 'exhaustive_stepwise':
            wf, models_transformed = self.algorithm(self.base_model, self.mfl)

            task_result = Task('results', post_process_results)
            wf.add_task(task_result, predecessors=models_transformed)

            res = self.dispatcher.run(wf, self.database)  # FIXME: postprocessing/collecting needed
        else:
            df = self.algorithm(
                self.base_model,
                self.mfl,
                self.fit,
                self.rankfunc,
            )
            res = ModelSearchResults(runs=df)
            res.to_json(path=self.rundir.path / 'results.json')
            res.to_csv(path=self.rundir.path / 'results.csv')
        return res


def post_process_results(*models):
    return models


class ModelSearchResults(pharmpy.results.Results):
    def __init__(self, runs=None):
        self.runs = runs


def run_modelsearch(base_model, algorithm, mfl, **kwargs):
    ms = ModelSearch(base_model, algorithm, mfl, **kwargs)
    res = ms.run()
    return res
