import functools

import pharmpy.execute as execute
import pharmpy.methods
import pharmpy.methods.modelfit as modelfit
import pharmpy.results
import pharmpy.search.algorithms as algorithms
import pharmpy.search.rankfuncs as rankfuncs


class ModelSearch(pharmpy.methods.Method):
    def __init__(self, base_model, modeling_funcs, **kwargs):
        self.base_model = base_model
        self.funcs = self.create_funcs_from_modeling(modeling_funcs)
        super().__init__(**kwargs)

    def create_funcs_from_modeling(self, modeling_funcs):
        """Create partial functions given a list of strings of function calls"""
        funcs = []
        import pharmpy.modeling

        for funcstr in modeling_funcs:
            name, args = funcstr.split('(')
            name = name.strip()
            args = args.strip()[:-1]
            if args:
                argdict = dict(
                    [e.split('=')[0].strip(), e.split('=')[1].strip()] for e in args.split(',')
                )
            else:
                argdict = dict()
            function = getattr(pharmpy.modeling, name)
            func = functools.partial(function, **argdict)
            funcs.append(func)
        return funcs

    def fit(self, models):
        db = execute.LocalDirectoryDatabase(self.rundir.path / 'models')
        modelfit_run = modelfit.Modelfit(models, database=db, path=self.rundir.path)
        modelfit_run.run()

    def run(self):
        df = algorithms.exhaustive(
            self.base_model,
            self.funcs,
            self.fit,
            rankfuncs.ofv,
        )
        res = ModelSearchResults(runs=df)
        res.to_json(path=self.rundir.path / 'results.json')
        res.to_csv(path=self.rundir.path / 'results.csv')
        return res


class ModelSearchResults(pharmpy.results.Results):
    def __init__(self, runs=None):
        self.runs = runs


def run_modelsearch(base_model, **kwargs):
    ms = ModelSearch(base_model, **kwargs)
    res = ms.run()
    return res
