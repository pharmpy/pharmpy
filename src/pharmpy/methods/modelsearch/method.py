import pharmpy.execute as execute
import pharmpy.methods
import pharmpy.methods.modelfit as modelfit
import pharmpy.results
import pharmpy.search.algorithms as algorithms
import pharmpy.search.rankfuncs as rankfuncs


class ModelSearch(pharmpy.methods.Method):
    def __init__(self, base_model, algorithm, mfl, rankfunc='ofv', **kwargs):
        self.base_model = base_model
        self.mfl = mfl
        self.algorithm = getattr(algorithms, algorithm)
        self.rankfunc = getattr(rankfuncs, rankfunc)
        super().__init__(**kwargs)

    def fit(self, models):
        db = execute.LocalDirectoryDatabase(self.rundir.path / 'models')
        modelfit_run = modelfit.Modelfit(models, database=db, path=self.rundir.path)
        modelfit_run.run()

    def run(self):
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


class ModelSearchResults(pharmpy.results.Results):
    def __init__(self, runs=None):
        self.runs = runs


def run_modelsearch(base_model, algorithm, mfl, **kwargs):
    ms = ModelSearch(base_model, algorithm, mfl, **kwargs)
    res = ms.run()
    return res
